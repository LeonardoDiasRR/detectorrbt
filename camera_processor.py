import cv2
import time
import logging
from pathlib import Path
from datetime import datetime
from threading import Thread, Lock
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from ultralytics import YOLO
from face_quality_score import get_face_quality_score

class CameraProcessor:
    def __init__(
        self,
        camera_id: int,
        camera_name: str,
        source: str,
        yolo_model: YOLO,
        device_type: str,
        batch_size: int,
        config: dict
    ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.source = source
        self.yolo_model = yolo_model
        self.device_type = device_type
        self.batch_size = batch_size
        self.config = config
        
        # Configurações do config.yaml
        self.max_frames_lost = config.get('max_frames_lost', 30)
        self.reconnect_interval = config.get('reconnect_interval', 5)
        self.max_reconnect_attempts = config.get('max_reconnect_attempts', 10)
        
        # Estruturas de dados para tracks
        self.active_tracks: Dict[int, List[Tuple[np.ndarray, float, datetime, tuple]]] = defaultdict(list)
        self.track_frames_lost: Dict[int, int] = defaultdict(int)
        self.lock = Lock()
        
        # Diretório de saída
        self.output_dir = Path('./imagens')
        self.output_dir.mkdir(exist_ok=True)
        
        # Logger específico para esta câmera
        self.logger = logging.getLogger(f"Camera_{camera_id}_{camera_name}")
        
        # Flag de execução
        self.running = False
        self.thread = None
        
    def start(self):
        """Inicia o processamento em uma thread separada"""
        self.running = True
        self.thread = Thread(target=self._process_stream, daemon=True)
        self.thread.start()
        self.logger.info(f"Processamento iniciado para câmera {self.camera_id} - {self.camera_name}")
        
    def stop(self):
        """Para o processamento"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.logger.info(f"Processamento finalizado para câmera {self.camera_id} - {self.camera_name}")
        
    def _process_stream(self):
        """Processa o stream de vídeo usando track nativo com batch"""
        attempts = 0
        
        while attempts < self.max_reconnect_attempts and self.running:
            try:
                self.logger.info(f"Tentando conectar ao stream (tentativa {attempts + 1}/{self.max_reconnect_attempts})")
                
                # Usa track nativo do YOLO com batch processing
                results = self.yolo_model.track(
                    source=self.source,
                    persist=True,
                    tracker="bytetrack.yaml",
                    device=self.device_type,
                    classes=[0],  # Apenas pessoa
                    verbose=False,
                    show=False,
                    stream=True,  # Streaming contínuo
                    batch=self.batch_size  # Batch processing
                )
                
                self.logger.info("Conexão estabelecida com sucesso")
                attempts = 0  # Reset tentativas após sucesso
                
                current_frame_tracks = set()
                frame_count = 0
                
                for result in results:
                    if not self.running:
                        break
                    
                    frame = result.orig_img.copy()
                    frame_count += 1
                    
                    # Reseta o conjunto de tracks visíveis no frame atual
                    if frame_count % self.batch_size == 1:
                        current_frame_tracks.clear()
                    
                    if result.boxes is None or len(result.boxes) == 0:
                        continue
                    
                    # Processa cada detecção com track_id
                    for box in result.boxes:
                        if box.id is None:
                            continue
                        
                        track_id = int(box.id[0])
                        current_frame_tracks.add(track_id)
                        
                        # Extrai coordenadas do bbox
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Extrai confiança da detecção
                        confidence = float(box.conf[0])
                        
                        # Extrai landmarks se disponíveis
                        landmarks = None
                        if hasattr(result, 'keypoints') and result.keypoints is not None:
                            try:
                                # Obtém índice do box atual
                                box_idx = int(box.id[0]) if hasattr(box.id, '__getitem__') else int(box.id)
                                if box_idx < len(result.keypoints.xy):
                                    kpts = result.keypoints.xy[box_idx]
                                    if len(kpts) >= 5:
                                        kpts_array = kpts[:5].cpu().numpy() if hasattr(kpts, 'cpu') else np.array(kpts[:5])
                                        landmarks = kpts_array if isinstance(kpts_array, np.ndarray) else None
                            except Exception:
                                pass
                        
                        # Calcula qualidade da face
                        quality_score = get_face_quality_score(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            frame=frame,
                            landmarks=landmarks
                        )
                        
                        # Armazena frame, qualidade e timestamp
                        timestamp = datetime.now()
                        
                        with self.lock:
                            self.active_tracks[track_id].append((frame.copy(), quality_score, timestamp, (x1, y1, x2, y2)))
                            self.track_frames_lost[track_id] = 0
                    
                    # Atualiza contadores de frames perdidos a cada batch
                    if frame_count % self.batch_size == 0:
                        self._update_lost_tracks(current_frame_tracks)
                
                # Se saiu do loop normalmente, finaliza todos os tracks
                self._finalize_all_tracks()
                break
                
            except KeyboardInterrupt:
                self.logger.info("Interrompido pelo usuário")
                break
                
            except Exception as e:
                attempts += 1
                self.logger.error(f"Erro no stream: {e}")
                
                if attempts < self.max_reconnect_attempts:
                    self.logger.info(f"Tentando reconectar em {self.reconnect_interval}s...")
                    time.sleep(self.reconnect_interval)
                else:
                    self.logger.error(f"Falha após {self.max_reconnect_attempts} tentativas")
                    break
        
        self._finalize_all_tracks()
        
    def _update_lost_tracks(self, current_frame_tracks: set):
        """Atualiza contadores de frames perdidos"""
        with self.lock:
            tracks_to_finalize = []
            
            for track_id in list(self.track_frames_lost.keys()):
                if track_id not in current_frame_tracks:
                    self.track_frames_lost[track_id] += 1
                    
                    if self.track_frames_lost[track_id] >= self.max_frames_lost:
                        tracks_to_finalize.append(track_id)
            
            # Finaliza tracks perdidos
            for track_id in tracks_to_finalize:
                self._finalize_track(track_id)
    
    def _finalize_track(self, track_id: int):
        """Finaliza um track e salva a melhor face"""
        with self.lock:
            if track_id not in self.active_tracks or len(self.active_tracks[track_id]) == 0:
                return
            
            # Encontra o frame com melhor qualidade
            best_frame, best_score, best_timestamp, best_bbox = max(
                self.active_tracks[track_id],
                key=lambda x: x[1]
            )
            
            # Salva o melhor frame
            self._save_best_frame(track_id, best_frame, best_score, best_timestamp, best_bbox)
            
            # Remove track da memória
            del self.active_tracks[track_id]
            del self.track_frames_lost[track_id]
            
            self.logger.info(f"Track {track_id} finalizado. Melhor score: {best_score:.4f}")
    
    def _save_best_frame(self, track_id: int, frame: np.ndarray, score: float, timestamp: datetime, bbox: tuple):
        """Salva o frame com bbox desenhado"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Desenha bbox verde
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Adiciona label com track_id e score
            label = f"Track {track_id} | Score: {score:.4f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Nome do arquivo
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"Camera_{self.camera_id}-{self.camera_name}_Track_{track_id}_{timestamp_str}.jpg"
            filepath = self.output_dir / filename
            
            # Salva com qualidade 95
            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            self.logger.info(f"Imagem salva: {filename}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar frame do track {track_id}: {e}", exc_info=True)
    
    def _finalize_all_tracks(self):
        """Finaliza todos os tracks ativos"""
        with self.lock:
            track_ids = list(self.active_tracks.keys())
        
        for track_id in track_ids:
            self._finalize_track(track_id)
        
        self.logger.info("Todos os tracks foram finalizados")