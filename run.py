# built-in
import os
import logging
from dotenv import load_dotenv
import traceback

# 3rd party
from ultralytics import YOLO
import torch

# local
from findface_multi import FindfaceMulti
from domain.adapters import FindfaceAdapter
from domain.services import ByteTrackDetectorService
from domain.entities import Camera
from domain.value_objects import IdVO, NameVO, CameraTokenVO, CameraSourceVO
from config_loader import CONFIG

# Configura logging ANTES de importar outros módulos
log_file = os.path.join(os.path.dirname(__file__), "detectorrbt.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()  # Mantém output no console também
    ],
    force=True  # Força reconfiguração mesmo se já foi chamado
)


def main():
    logger = logging.getLogger(__name__)
    
    # Limpa o diretório de imagens antes de iniciar
    imagens_dir = os.path.join(os.path.dirname(__file__), "imagens")
    if os.path.exists(imagens_dir):
        import shutil
        shutil.rmtree(imagens_dir)
        logger.info(f"Diretório '{imagens_dir}' limpo.")
    
    # Recria o diretório vazio
    os.makedirs(imagens_dir, exist_ok=True)
    logger.info(f"Diretório '{imagens_dir}' criado.")
    
    # Usa configurações do CONFIG
    gpu_index = CONFIG.get("gpu_index", 0)
    device_type = f"cuda:{gpu_index}" if (torch is not None and torch.cuda.is_available()) else "cpu"

    # Choose batch size based on GPU availability
    if device_type.startswith("cuda"):
        batch = CONFIG.get("gpu_batch_size", 32)
    else:
        batch = CONFIG.get("cpu_batch_size", 4)

    show = CONFIG.get("show", True)
    cameras = CONFIG.get("cameras", [])
    processors = []

    # Prepare a YOLO model instance (shared)
    try:
        model_path = CONFIG.get("face_detection_model", "yolov8n-face.pt")
        yolo_model = YOLO(model_path)
        logger.info(f"Modelo YOLO carregado: {model_path} no dispositivo {device_type}")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo YOLO: {e}")
        yolo_model = None
    
    # Cria adapter do FindFace
    findface_adapter = FindfaceAdapter(
        ff, 
        camera_prefix=CONFIG.get("prefixo_grupo_camera_findface", "TESTE")
    )
    
    # Obtém câmeras usando o adapter
    cameras_ff = findface_adapter.get_cameras()
    logger.info(f"Total de câmeras obtidas: {len(cameras_ff)}")
    
    # Adiciona câmeras extras do config (se houver)
    for cam in cameras:
        camera = Camera(
            camera_id=IdVO(cam.get("id", 0)),
            camera_name=NameVO(cam.get("name", "Camera Local")),
            camera_token=CameraTokenVO(cam.get("token", "")),
            source=CameraSourceVO(cam.get("url", ""))
        )
        cameras_ff.append(camera)
    
    for camera in cameras_ff:
        # Cria serviço de detecção usando entidades do domínio e o adapter
        processor = ByteTrackDetectorService(
            camera=camera,
            yolo_model=yolo_model,
            findface_adapter=findface_adapter,
            tracker=CONFIG.get("tracker", "bytetrack.yaml"),
            batch=batch,
            show=show,
            conf=CONFIG.get("conf", 0.1),
            iou=CONFIG.get("iou", 0.2),
            max_frames_lost=CONFIG.get("max_frames_lost", 30),
            verbose_log=CONFIG.get("verbose_log", False)
        )
        processors.append(processor)

    try:
        for proc in processors:
            proc.start()
    except KeyboardInterrupt:
        logger.info("Interrupção detectada. Finalizando...")
        for proc in processors:
            proc.stop()


if __name__ == "__main__":
    # Carrega variáveis de ambiente do arquivo .env
    load_dotenv()

    try:
        ff = FindfaceMulti(		
            url_base=os.environ["FINDFACE_URL"],
            user=os.environ["FINDFACE_USER"],
            password=os.environ["FINDFACE_PASSWORD"],
            uuid=os.environ["FINDFACE_UUID"]
        )
        main()
        
    except KeyError as e:
        print(traceback.format_exc())
    finally:
        ff.logout()