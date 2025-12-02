from typing import Optional
from ultralytics import YOLO
import cv2
import time
import logging
from threading import Thread


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
            ):
        self.source = source
        self.model = yolo_model
        self.project = project
        self.name = name
        self.tracker = tracker
        self.batch = batch
        self.show = show
        self.stream = stream
        self.conf = 0.3
        self.iou = 0.5
        self.running = False
        self.thread = None
        self.logger = logging.getLogger(f"ByteTrackDetector_{camera_id}_{camera_name}")

    def start(self):
        """Inicia o processamento em uma thread separada"""
        self.running = True
        self.thread = Thread(target=self._process_stream, daemon=True)
        self.thread.start()
        self.logger.info(f"ByteTrackDetector iniciado")

    def stop(self):
        """Para o processamento"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.logger.info(f"ByteTrackDetector finalizado")

    def join(self, timeout=None):
        """Aguarda a thread finalizar"""
        if self.thread:
            self.thread.join(timeout=timeout)

    def _process_stream(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
        while self.running:
            try:
                # Usar stream=True para fontes contínuas e permitir reconexão em caso de falha
                for _res in self.model.track(
                    source=self.source,
                    tracker=self.tracker,  # Usar ByteTrack para rastreamento
                    persist=True,             # Manter IDs de rastreamento consistentes
                    conf=self.conf,           # Limite de confiança
                    iou=self.iou,             # Limite de IoU
                    show=self.show,           # Mostrar a saída em tempo real (requer OpenCV)
                    stream=self.stream,       # ITERAR sobre frames contínuos
                    batch=self.batch
                ):
                    if not self.running:
                        break
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
    detector = ByteTrackDetector(source=RTSP_URL)
    detector.start_detection()