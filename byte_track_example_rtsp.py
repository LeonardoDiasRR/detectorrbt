from ultralytics import YOLO
import cv2
import time
import logging

# Definir o URL do fluxo RTSP
# Formato típico: "rtsp://username:password@ip_address:port/path_to_stream"
# Exemplo genérico abaixo. Substitua pelos seus dados reais.
RTSP_URL = "rtsp://findface:Mitra2021@10.95.7.24:7001/022cdcf4-b31d-f379-c2d8-b4e727f8a148"

# Carregar um modelo pré-treinado (YOLOv8nano)
model = YOLO("yolov8n-face.pt")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    while True:
        try:
            # Usar stream=True para fontes contínuas e permitir reconexão em caso de falha
            for _res in model.track(
                source=RTSP_URL,
                tracker="bytetrack.yaml",  # Usar ByteTrack para rastreamento
                persist=True,             # Manter IDs de rastreamento consistentes
                conf=0.3,                 # Limite de confiança
                iou=0.5,                  # Limite de IoU
                show=True,                # Mostrar a saída em tempo real (requer OpenCV)
                stream=True,              # ITERAR sobre frames contínuos
                save_frames=True,
                save_txt=True,
                save_crop=True,
                project="./imagens/",
                name="rtsp_byte_track_results",
                batch=4
            ):
                pass
        except KeyboardInterrupt:
            logging.info("Execução interrompida pelo usuário.")
            break
        except Exception:
            logging.exception("Erro no stream RTSP. Tentando reconectar em 5 segundos...")
            time.sleep(5)

if __name__ == "__main__":
    main()
