from typing import Optional, Dict, List, Tuple
from collections import defaultdict
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import cv2
import time
import logging
from face_quality_score import get_face_quality_score
from track_processor import TrackProcessor
from findface_multi import FindfaceMulti


class ByteTrackDetector:
    def __init__(
            self,
            camera_id: int,
            camera_name: str,
            camera_token: str,
            source: str, 
            yolo_model: Optional[YOLO],
            findface: Optional[FindfaceMulti] = None,
            project="./imagens/", 
            name="rtsp_byte_track_results", 
            tracker="bytetrack.yaml",
            batch=4,
            show=True,
            stream=True,
            conf=0.1,
            iou=0.2,
            max_frames_lost=30,
            verbose_log=False
            ):
        # Suprime warnings do OpenCV
        cv2.setLogLevel(0)
        
        self.source = source
        self.model = yolo_model
        self.project = project
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.name = name
        self.camera_token = camera_token
        self.tracker = tracker
        self.batch = batch
        self.show = show
        self.stream = stream
        self.conf = conf
        self.iou = iou
        self.running = False
        self.max_frames_lost = max_frames_lost
        self.verbose_log = verbose_log
        self.findface = findface  # Armazena instância do FindFace
        self.logger = logging.getLogger(f"ByteTrackDetector_{camera_id}_{camera_name}")
        
        # Estruturas para rastreamento de tracks
        self.active_tracks: Dict[int, List[Tuple[np.ndarray, float, datetime, Tuple, Optional[np.ndarray]]]] = defaultdict(list)
        self.track_frames_lost: Dict[int, int] = defaultdict(int)

    def start(self):
        """Inicia o processamento diretamente (sem thread interna)"""
        self.running = True
        self.logger.info(f"ByteTrackDetector iniciado")
        self._process_stream()

    def stop(self):
        """Para o processamento"""
        self.running = False
        self.logger.info(f"ByteTrackDetector finalizado")

    def join(self, timeout=None):
        """Método mantido para compatibilidade (não faz nada pois não há thread interna)"""
        pass

    def _process_stream(self):
        while self.running:
            try:
                for result in self.model.track(
                    source=self.source,
                    tracker=self.tracker,
                    persist=True,
                    conf=self.conf,
                    iou=self.iou,
                    show=self.show,
                    stream=self.stream,
                    batch=self.batch,
                    verbose=False  # Suprime mensagens de inferência do YOLO
                ):
                    if not self.running:
                        break
                    
                    frame = result.orig_img
                    current_frame_tracks = set()
                    
                    # Processa detecções do frame atual
                    if result.boxes is not None and result.boxes.id is not None:
                        for i, box in enumerate(result.boxes):
                            track_id = int(box.id[0])
                            current_frame_tracks.add(track_id)
                            
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            
                            # Landmarks
                            landmarks = None
                            if result.keypoints is not None and len(result.keypoints) > i:
                                kpts = result.keypoints[i].xy[0].cpu().numpy()
                                landmarks = kpts
                            
                            # Calcula qualidade
                            bbox_tuple = (x1, y1, x2, y2)
                            quality_score = get_face_quality_score(
                                bbox=bbox_tuple,
                                confidence=confidence,
                                frame=frame,
                                landmarks=landmarks
                            )
                            
                            # Armazena dados do track
                            timestamp = datetime.now()
                            self.active_tracks[track_id].append(
                                (frame, quality_score, timestamp, bbox_tuple, landmarks, confidence))
                            # Reseta contador de frames perdidos
                            self.track_frames_lost[track_id] = 0
                            
                            if self.verbose_log:
                                self.logger.info(
                                    f"Track {track_id}: bbox({x1},{y1},{x2},{y2}), "
                                    f"conf={confidence:.2f}, quality_score={quality_score:.4f}"
                                )
                    
                    # Atualiza tracks perdidos
                    self._update_lost_tracks(current_frame_tracks)
                    # Atualiza tracks perdidos
                    self._update_lost_tracks(current_frame_tracks)

            except KeyboardInterrupt:
                self.logger.info("Execução interrompida pelo usuário.")
                break
            except Exception:
                self.logger.exception("Erro no stream RTSP. Tentando reconectar em 5 segundos...")
                time.sleep(5)
        
        # Finaliza todos os tracks restantes ao encerrar
        self._finalize_all_tracks()

    def _update_lost_tracks(self, current_frame_tracks: set):
        """Atualiza contadores de frames perdidos e finaliza tracks"""
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
        """Finaliza um track e processa a melhor face"""
        if track_id not in self.active_tracks or len(self.active_tracks[track_id]) == 0:
            return
        
        self.logger.info(f"Track {track_id} finalizado após {self.track_frames_lost[track_id]} frames perdidos")
        
        # Cria processador para o track
        processor = TrackProcessor(
            track_id=track_id,
            detections=self.active_tracks[track_id],
            camera_id=self.camera_id,
            camera_name=self.camera_name,
            camera_token=self.camera_token,
            findface=self.findface,
            project=self.project,
            name=self.name
        )
        
        # Remove track da memória ANTES de processar
        # Isso previne processamento duplicado se o track_id for reutilizado
        track_data = self.active_tracks[track_id]
        del self.active_tracks[track_id]
        del self.track_frames_lost[track_id]
        
        # Processa o track (encontra melhor face, salva e envia para FindFace)
        processor.process()

    def _finalize_all_tracks(self):
        """Finaliza todos os tracks ativos"""
        track_ids = list(self.active_tracks.keys())
        for track_id in track_ids:
            self._finalize_track(track_id)
        self.logger.info("Todos os tracks foram finalizados")

if __name__ == "__main__":
    # Definir o URL do fluxo RTSP
    # Formato típico: "rtsp://username:password@ip_address:port/path_to_stream"
    # Exemplo genérico abaixo. Substitua pelos seus dados reais.
    RTSP_URL = "rtsp://findface:Mitra2021@10.95.7.24:7001/022cdcf4-b31d-f379-c2d8-b4e727f8a148"
    detector = ByteTrackDetector(source=RTSP_URL, camera_id=1, camera_name="Camera_RTSP", yolo_model=YOLO("yolov8n-face.pt"), show=True)
    detector.start()