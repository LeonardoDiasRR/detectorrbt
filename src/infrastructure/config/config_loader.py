"""
Carregador de configurações.
Responsável por ler arquivos YAML e variáveis de ambiente,
convertendo-os em objetos de configuração type-safe.
"""

import os
import yaml
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from .settings import (
    AppSettings,
    FindFaceConfig,
    YOLOConfig,
    ByteTrackConfig,
    ProcessingConfig,
    StorageConfig,
    CameraConfig
)


class ConfigLoader:
    """Carrega configurações de arquivos e variáveis de ambiente."""
    
    @staticmethod
    def load_from_yaml(yaml_path: str = "config.yaml") -> dict:
        """
        Carrega configurações de arquivo YAML.
        
        :param yaml_path: Caminho para o arquivo YAML.
        :return: Dicionário com configurações.
        """
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {yaml_path}")
        
        with open(yaml_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    @staticmethod
    def load_from_env() -> FindFaceConfig:
        """
        Carrega configurações do FindFace de variáveis de ambiente.
        
        :return: Configuração do FindFace.
        :raises ValueError: Se variáveis obrigatórias não estiverem definidas.
        """
        load_dotenv()
        
        required_vars = ["FINDFACE_URL", "FINDFACE_USER", "FINDFACE_PASSWORD", "FINDFACE_UUID"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Variáveis de ambiente obrigatórias não definidas: {', '.join(missing_vars)}")
        
        return FindFaceConfig(
            url_base=os.getenv("FINDFACE_URL"),
            user=os.getenv("FINDFACE_USER"),
            password=os.getenv("FINDFACE_PASSWORD"),
            uuid=os.getenv("FINDFACE_UUID")
        )
    
    @classmethod
    def load(cls, yaml_path: str = "config.yaml") -> AppSettings:
        """
        Carrega todas as configurações da aplicação.
        
        :param yaml_path: Caminho para o arquivo YAML.
        :return: Objeto AppSettings completo.
        """
        # Carrega do YAML
        yaml_config = cls.load_from_yaml(yaml_path)
        
        # Carrega FindFace do .env
        findface_config = cls.load_from_env()
        
        # Adiciona prefixo de câmera do YAML ao FindFace
        findface_config.camera_prefix = yaml_config.get("prefixo_grupo_camera_findface", "EXTERNO")
        
        # Monta configurações
        yolo_config = YOLOConfig(
            model_path=yaml_config.get("face_detection_model", "yolov8n-face.pt"),
            conf_threshold=yaml_config.get("conf", 0.1),
            iou_threshold=yaml_config.get("iou", 0.2)
        )
        
        bytetrack_config = ByteTrackConfig(
            tracker_config=yaml_config.get("tracker", "bytetrack.yaml"),
            max_frames_lost=yaml_config.get("max_frames_lost", 30)
        )
        
        processing_config = ProcessingConfig(
            gpu_index=yaml_config.get("gpu_index", 0),
            gpu_batch_size=yaml_config.get("gpu_batch_size", 32),
            cpu_batch_size=yaml_config.get("cpu_batch_size", 4),
            show_video=yaml_config.get("show", True),
            verbose_log=yaml_config.get("verbose_log", False)
        )
        
        storage_config = StorageConfig(
            project_dir=yaml_config.get("project", "./imagens/"),
            results_dir=yaml_config.get("name", "rtsp_byte_track_results")
        )
        
        # Carrega câmeras do YAML
        cameras = [
            CameraConfig(
                id=cam.get("id", 0),
                name=cam.get("name", "Camera Local"),
                url=cam.get("url", ""),
                token=cam.get("token", "")
            )
            for cam in yaml_config.get("cameras", [])
        ]
        
        return AppSettings(
            findface=findface_config,
            yolo=yolo_config,
            bytetrack=bytetrack_config,
            processing=processing_config,
            storage=storage_config,
            cameras=cameras
        )