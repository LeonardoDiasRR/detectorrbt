"""
Serviço de detecção e rastreamento de faces usando ByteTrack no domínio.
"""

# built-in
from typing import Optional, Dict, List
from collections import defaultdict
from datetime import datetime
import logging
import time

# 3rd party
import numpy as np
import cv2
from ultralytics import YOLO

# local
from src.domain.adapters.findface_adapter import FindfaceAdapter
from src.domain.entities import Camera, Frame, Event, Track
from src.domain.value_objects import IdVO, BboxVO, ConfidenceVO, LandmarksVO, TimestampVO


class ByteTrackDetectorService:
    """
    Serviço de domínio responsável por detectar e rastrear faces em streams de vídeo.
    Utiliza entidades de domínio (Camera, Frame, Event, Track) seguindo princípios DDD.
    """

    def __init__(
        self,
        camera: Camera,
        yolo_model: YOLO,
        findface_adapter: Optional[FindfaceAdapter] = None,
        tracker: str = "bytetrack.yaml",
        batch: int = 4,
        show: bool = True,
        stream: bool = True,
        conf: float = 0.1,
        iou: float = 0.2,
        max_frames_lost: int = 30,
        verbose_log: bool = False,
        project_dir: str = "./imagens/",
        results_dir: str = "rtsp_byte_track_results",
        min_movement_threshold: float = 50.0,
        min_movement_percentage: float = 0.3
    ):
        """
        Inicializa o serviço de detecção de faces.

        :param camera: Entidade Camera com informações da câmera.
        :param yolo_model: Modelo YOLO para detecção de faces.
        :param findface_adapter: Adapter para comunicação com FindFace (opcional).
        :param tracker: Arquivo de configuração do tracker ByteTrack.
        :param batch: Tamanho do batch para processamento.
        :param show: Se deve exibir o vídeo processado.
        :param stream: Se deve processar em modo stream.
        :param conf: Threshold de confiança para detecções.
        :param iou: Threshold de IOU para o tracker.
        :param max_frames_lost: Máximo de frames perdidos antes de finalizar um track.
        :param verbose_log: Se deve exibir logs detalhados.
        :param project_dir: Diretório base para salvamento de imagens.
        :param results_dir: Nome do subdiretório para resultados.
        :param min_movement_threshold: Limite mínimo de movimento em pixels.
        :param min_movement_percentage: Percentual mínimo de frames com movimento.
        :raises TypeError: Se camera não for do tipo Camera.
        """
        if not isinstance(camera, Camera):
            raise TypeError(f"camera deve ser Camera, recebido: {type(camera).__name__}")
        
        if findface_adapter is not None and not isinstance(findface_adapter, FindfaceAdapter):
            raise TypeError(f"findface_adapter deve ser FindfaceAdapter, recebido: {type(findface_adapter).__name__}")
        
        # Suprime warnings do OpenCV
        cv2.setLogLevel(0)
        
        self.camera = camera
        self.model = yolo_model
        self.findface_adapter = findface_adapter
        self.tracker = tracker
        self.batch = batch
        self.show = show
        self.stream = stream
        self.conf = conf
        self.iou = iou
        self.max_frames_lost = max_frames_lost
        self.verbose_log = verbose_log
        self.project_dir = project_dir
        self.results_dir = results_dir
        self.min_movement_threshold = min_movement_threshold
        self.min_movement_percentage = min_movement_percentage
        self.running = False
        
        self.logger = logging.getLogger(
            f"ByteTrackDetectorService_{camera.camera_id.value()}_{camera.camera_name.value()}"
        )
        
        # Estruturas para rastreamento de tracks usando entidades do domínio
        self.active_tracks: Dict[int, Track] = {}
        self.track_frames_lost: Dict[int, int] = defaultdict(int)
        
        # Contador global de IDs para frames e eventos
        self._frame_id_counter = 0
        self._event_id_counter = 0

    def start(self):
        """Inicia o processamento do stream de vídeo"""
        self.running = True
        self.logger.info(
            f"ByteTrackDetectorService iniciado para câmera "
            f"{self.camera.camera_name.value()} (ID: {self.camera.camera_id.value()})"
        )
        self._process_stream()

    def stop(self):
        """Para o processamento do stream"""
        self.running = False
        self.logger.info(
            f"ByteTrackDetectorService finalizado para câmera "
            f"{self.camera.camera_name.value()}"
        )

    def _process_stream(self):
        """Processa o stream de vídeo frame a frame"""
        while self.running:
            try:
                for result in self.model.track(
                    source=self.camera.source.value(),
                    tracker=self.tracker,
                    persist=True,
                    conf=self.conf,
                    iou=self.iou,
                    show=self.show,
                    stream=self.stream,
                    batch=self.batch,
                    verbose=False
                ):
                    if not self.running:
                        break
                    
                    # Cria entidade Frame
                    frame_entity = self._create_frame_entity(result.orig_img)
                    current_frame_tracks = set()
                    
                    # Processa detecções do frame atual
                    if result.boxes is not None and result.boxes.id is not None:
                        for i, box in enumerate(result.boxes):
                            track_id = int(box.id[0])
                            current_frame_tracks.add(track_id)
                            
                            # Cria Event para esta detecção
                            event = self._create_event_from_detection(
                                frame_entity, box, result.keypoints, i
                            )
                            
                            # Adiciona evento ao track
                            if track_id not in self.active_tracks:
                                self.active_tracks[track_id] = Track(
                                    id=IdVO(track_id),
                                    events=[]
                                )
                            
                            self.active_tracks[track_id].add_event(event)
                            self.track_frames_lost[track_id] = 0
                            
                            if self.verbose_log:
                                self.logger.info(
                                    f"Track {track_id}: bbox={event.bbox.value()}, "
                                    f"conf={event.confidence.value():.2f}, "
                                    f"quality={event.face_quality_score.value():.4f}"
                                )
                    
                    # Atualiza tracks perdidos
                    self._update_lost_tracks(current_frame_tracks)

            except KeyboardInterrupt:
                self.logger.info("Execução interrompida pelo usuário.")
                break
            except Exception as e:
                self.logger.exception(f"Erro no stream RTSP: {e}. Tentando reconectar em 5 segundos...")
                time.sleep(5)
        
        # Finaliza todos os tracks restantes ao encerrar
        self._finalize_all_tracks()

    def _create_frame_entity(self, frame_array: np.ndarray) -> Frame:
        """
        Cria uma entidade Frame a partir de um numpy array.
        
        :param frame_array: Array numpy do frame.
        :return: Entidade Frame.
        """
        self._frame_id_counter += 1
        return Frame(
            id=IdVO(self._frame_id_counter),
            ndarray=frame_array,
            camera_id=self.camera.camera_id,
            camera_name=self.camera.camera_name,
            camera_token=self.camera.camera_token,
            timestamp=TimestampVO.now()
        )

    def _create_event_from_detection(
        self,
        frame: Frame,
        box,
        keypoints,
        index: int
    ) -> Event:
        """
        Cria uma entidade Event a partir de uma detecção YOLO.
        
        :param frame: Entidade Frame onde a detecção ocorreu.
        :param box: Box da detecção YOLO.
        :param keypoints: Keypoints da detecção.
        :param index: Índice da detecção no frame.
        :return: Entidade Event.
        """
        self._event_id_counter += 1
        
        # Extrai bbox
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = BboxVO((x1, y1, x2, y2))
        
        # Extrai confiança
        confidence = ConfidenceVO(float(box.conf[0]))
        
        # Extrai landmarks
        landmarks_array = None
        if keypoints is not None and len(keypoints) > index:
            kpts = keypoints[index].xy[0].cpu().numpy()
            landmarks_array = kpts
        landmarks = LandmarksVO(landmarks_array)
        
        # Cria evento (o face_quality_score é calculado automaticamente)
        event = Event(
            id=IdVO(self._event_id_counter),
            frame=frame,
            bbox=bbox,
            confidence=confidence,
            landmarks=landmarks
        )
        
        return event

    def _update_lost_tracks(self, current_frame_tracks: set):
        """
        Atualiza contadores de frames perdidos e finaliza tracks.
        
        :param current_frame_tracks: Set com IDs dos tracks presentes no frame atual.
        """
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
        """
        Finaliza um track: encontra melhor evento, salva e envia para FindFace.
        
        :param track_id: ID do track a ser finalizado.
        """
        if track_id not in self.active_tracks:
            return
        
        track = self.active_tracks[track_id]
        
        if track.is_empty:
            self.logger.warning(f"Track {track_id} vazio, não será processado")
            del self.active_tracks[track_id]
            del self.track_frames_lost[track_id]
            return
        
        # Verifica se houve movimento
        has_movement = track.has_movement(
            min_threshold_pixels=self.min_movement_threshold,
            min_frame_percentage=self.min_movement_percentage
        )
        
        # Obtém estatísticas de movimento
        movement_stats = track.get_movement_statistics()
        
        self.logger.info(
            f"Track {track_id} finalizado após {self.track_frames_lost[track_id]} frames perdidos. "
            f"Total de eventos: {track.event_count} | "
            f"Movimento detectado: {has_movement} | "
            f"Distância média: {movement_stats['average_distance']:.2f}px | "
            f"Distância máxima: {movement_stats['max_distance']:.2f}px"
        )
        
        # Remove track da memória ANTES de processar
        del self.active_tracks[track_id]
        del self.track_frames_lost[track_id]
        
        # Obtém melhor evento do track
        best_event = track.get_best_event()
        
        if best_event is None:
            self.logger.warning(f"Track {track_id} não possui melhor evento")
            return
        
        # Salva melhor face
        self._save_best_event(track_id, best_event, track.event_count, has_movement)
        
        # Envia para FindFace apenas se houver movimento
        if self.findface_adapter is not None and has_movement:
            self._send_best_event_to_findface(track_id, best_event, track.event_count)
        elif not has_movement:
            self.logger.info(f"Track {track_id} descartado: sem movimento significativo")

    def _save_best_event(self, track_id: int, event: Event, total_events: int, has_movement: bool):
        """
        Salva o melhor evento do track em disco com bbox desenhado.
        
        :param track_id: ID do track.
        :param event: Melhor evento do track.
        :param total_events: Total de eventos no track.
        :param has_movement: Se o track teve movimento significativo.
        """
        try:
            # Cria uma cópia do frame para desenhar
            frame_with_bbox = event.frame.ndarray.copy()
            
            x1, y1, x2, y2 = event.bbox.value()
            
            # Cor do bbox: verde se houver movimento, amarelo se não houver
            bbox_color = (0, 255, 0) if has_movement else (0, 255, 255)
            
            # Desenha bbox
            cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), bbox_color, 2)
            
            # Label
            movement_label = "MOV" if has_movement else "STATIC"
            label = (
                f"Track {track_id} | "
                f"{movement_label} | "
                f"Quality: {event.face_quality_score.value():.4f} | "
                f"Conf: {event.confidence.value():.2f}"
            )
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame_with_bbox,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                bbox_color,
                -1
            )
            cv2.putText(
                frame_with_bbox,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
            
            # Nome do arquivo
            timestamp_str = event.frame.timestamp.value().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            movement_prefix = "MOV" if has_movement else "STATIC"
            filename = f"{movement_prefix}_Track_{track_id}_{timestamp_str}.jpg"
            
            # Diretório
            from pathlib import Path
            filepath = Path(self.project_dir) / self.results_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Salva
            cv2.imwrite(str(filepath), frame_with_bbox, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            self.logger.info(
                f"Melhor face salva: {filename} "
                f"(quality={event.face_quality_score.value():.4f}, "
                f"conf={event.confidence.value():.2f}) | "
                f"Total de eventos: {total_events}"
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar evento do track {track_id}: {e}", exc_info=True)

    def _send_best_event_to_findface(self, track_id: int, event: Event, total_events: int):
        """
        Envia o melhor evento do track para o FindFace usando o adapter.
        
        :param track_id: ID do track.
        :param event: Melhor evento do track.
        :param total_events: Total de eventos no track.
        """
        try:
            # Envia para FindFace usando o adapter
            resposta = self.findface_adapter.send_event(event)
            
            # Verifica sucesso
            if resposta:
                self.logger.info(
                    f"✓ FindFace - Envio BEM-SUCEDIDO - Track {track_id}: "
                    f"quality={event.face_quality_score.value():.4f}, "
                    f"total_events={total_events}, "
                    f"camera_id={event.camera_id.value()}, "
                    f"resposta={resposta}"
                )
            else:
                self.logger.warning(
                    f"✗ FindFace - Envio retornou resposta vazia - Track {track_id}: "
                    f"quality={event.face_quality_score.value():.4f}, "
                    f"total_events={total_events}, "
                    f"camera_id={event.camera_id.value()}"
                )
                
        except Exception as e:
            self.logger.error(
                f"✗ FindFace - FALHA no envio - Track {track_id}: "
                f"quality={event.face_quality_score.value():.4f}, "
                f"total_events={total_events}, "
                f"camera_id={event.camera_id.value()}, "
                f"erro={e}",
                exc_info=True
            )

    def _finalize_all_tracks(self):
        """Finaliza todos os tracks ativos"""
        track_ids = list(self.active_tracks.keys())
        for track_id in track_ids:
            self._finalize_track(track_id)
        self.logger.info("Todos os tracks foram finalizados")
