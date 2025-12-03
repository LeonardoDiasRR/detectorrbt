from typing import Optional
from ultralytics import YOLO
import cv2
import time
import logging
from face_quality_score import get_face_quality_score


class ByteTrackDetector:
    def __init__(
            self,
            camera_id: int,
            camera_name: str,
            source: str, 
            yolo_model: Optional[YOLO], 
            project="./imagens/", 
            name="rtsp_byte_track_results", 
            tracker="bytetrack.yaml",
            batch=4,
            show=True,
            stream=True,
            conf=0.3,
            iou=0.5
            ):
        self.source = source
        self.model = yolo_model
        self.project = project
        self.name = name
        self.tracker = tracker
        self.batch = batch
        self.show = show
        self.stream = stream
        self.conf = conf
        self.iou = iou
        self.running = False
        self.logger = logging.getLogger(f"ByteTrackDetector_{camera_id}_{camera_name}")

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
                # Usar stream=True para fontes contínuas e permitir reconexão em caso de falha
                for result in self.model.track(
                    source=self.source,
                    tracker=self.tracker, # Usar ByteTrack para rastreamento
                    persist=True,  # Manter IDs de rastreamento consistentes
                    conf=self.conf,  # Limite de confiança
                    iou=self.iou,  # Limite de IoU
                    show=self.show,  # Mostrar a saída em tempo real (requer OpenCV)
                    stream=self.stream,  # ITERAR sobre frames contínuos
                    batch=self.batch
                ):
                    if not self.running:
                        break
                    
                    # Acessa o frame original
                    frame = result.orig_img  # numpy array (H, W, C)
                    
                    # Acessa as detecções com track IDs
                    if result.boxes is not None and result.boxes.id is not None:
                        for i, box in enumerate(result.boxes):
                            track_id = int(box.id[0])  # Track ID do ByteTrack
                            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas do bbox
                            confidence = float(box.conf[0])  # Confiança da detecção
                            class_id = int(box.cls[0])  # ID da classe
                            
                            # Acessa landmarks faciais (se disponíveis)
                            landmarks = None
                            if result.keypoints is not None and len(result.keypoints) > i:
                                # Keypoints vem como tensor [N, num_keypoints, 2 ou 3]
                                # onde N é o número de detecções
                                # Para yolov8n-face.pt geralmente são 5 pontos: [olho_esq, olho_dir, nariz, boca_esq, boca_dir]
                                kpts = result.keypoints[i].xy[0].cpu().numpy()  # [num_keypoints, 2]
                                landmarks = kpts
                            
                            # Calcula o score de qualidade da face
                            bbox_tuple = (x1, y1, x2, y2)
                            quality_score = get_face_quality_score(
                                bbox=bbox_tuple,
                                confidence=confidence,
                                frame=frame,
                                landmarks=landmarks
                            )
                            
                            # Crop da face do frame
                            face_crop = frame[y1:y2, x1:x2]
                            
                            self.logger.info(
                                f"Track {track_id}: bbox({x1},{y1},{x2},{y2}), "
                                f"conf={confidence:.2f}, quality_score={quality_score:.4f}"
                            )

            except KeyboardInterrupt:
                self.logger.info("Execução interrompida pelo usuário.")
                break
            except Exception:
                self.logger.exception("Erro no stream RTSP. Tentando reconectar em 5 segundos...")
                time.sleep(5)

if __name__ == "__main__":
    # Definir o URL do fluxo RTSP
    # Formato típico: "rtsp://username:password@ip_address:port/path_to_stream"
    # Exemplo genérico abaixo. Substitua pelos seus dados reais.
    RTSP_URL = "rtsp://findface:Mitra2021@10.95.7.24:7001/022cdcf4-b31d-f379-c2d8-b4e727f8a148"
    detector = ByteTrackDetector(source=RTSP_URL, camera_id=1, camera_name="Camera_RTSP", yolo_model=YOLO("yolov8n-face.pt"), show=True)
    detector.start()