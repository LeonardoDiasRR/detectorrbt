# built-in
import os
import logging
import traceback

# 3rd party
from ultralytics import YOLO

# local
from src.infrastructure import ConfigLoader, AppSettings
from src.infrastructure.external.findface_client import create_findface_client
from src.domain.adapters import FindfaceAdapter
from src.domain.services import ByteTrackDetectorService
from src.domain.entities import Camera
from src.domain.value_objects import IdVO, NameVO, CameraTokenVO, CameraSourceVO

# Configura logging
log_file = os.path.join(os.path.dirname(__file__), "detectorrbt.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)


def main(settings: AppSettings, findface_adapter: FindfaceAdapter):
    logger = logging.getLogger(__name__)
    
    # Limpa o diretório de imagens antes de iniciar
    imagens_dir = os.path.join(os.path.dirname(__file__), settings.storage.project_dir)
    if os.path.exists(imagens_dir):
        import shutil
        shutil.rmtree(imagens_dir)
        logger.info(f"Diretório '{imagens_dir}' limpo.")
    
    os.makedirs(imagens_dir, exist_ok=True)
    logger.info(f"Diretório '{imagens_dir}' criado.")
    
    processors = []

    # Carrega modelo YOLO
    try:
        yolo_model = YOLO(settings.yolo.model_path)
        logger.info(f"Modelo YOLO carregado: {settings.yolo.model_path} no dispositivo {settings.device}")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo YOLO: {e}")
        return
    
    # Obtém câmeras usando o adapter
    cameras_ff = findface_adapter.get_cameras()
    logger.info(f"Total de câmeras obtidas do FindFace: {len(cameras_ff)}")
    
    # Adiciona câmeras extras do config
    for cam_config in settings.cameras:
        camera = Camera(
            camera_id=IdVO(cam_config.id),
            camera_name=NameVO(cam_config.name),
            camera_token=CameraTokenVO(cam_config.token),
            source=CameraSourceVO(cam_config.url)
        )
        cameras_ff.append(camera)
    
    # Cria serviços de detecção
    for camera in cameras_ff:
        processor = ByteTrackDetectorService(
            camera=camera,
            yolo_model=yolo_model,
            findface_adapter=findface_adapter,
            tracker=settings.bytetrack.tracker_config,
            batch=settings.batch_size,
            show=settings.processing.show_video,
            conf=settings.yolo.conf_threshold,
            iou=settings.yolo.iou_threshold,
            max_frames_lost=settings.bytetrack.max_frames_lost,
            verbose_log=settings.processing.verbose_log,
            project_dir=settings.storage.project_dir,
            results_dir=settings.storage.results_dir,
            min_movement_threshold=settings.movement.min_movement_threshold_pixels,
            min_movement_percentage=settings.movement.min_movement_frame_percentage,
            min_confidence_threshold=settings.validation.min_confidence,
            max_frames_per_track=settings.bytetrack.max_frames_per_track  # ATUALIZADO
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
    try:
        # Carrega configurações type-safe
        settings = ConfigLoader.load()
        
        # Cria cliente FindFace
        ff = create_findface_client(settings.findface)
        
        # Cria adapter do FindFace
        findface_adapter = FindfaceAdapter(ff, camera_prefix=settings.findface.camera_prefix)
        
        # Executa aplicação
        main(settings, findface_adapter)
        
    except Exception as e:
        print(f"Erro ao iniciar aplicação: {e}")
        print(traceback.format_exc())
    finally:
        if 'ff' in locals():
            ff.logout()