# src/infrastructure/model/model_factory.py
"""
Factory para criação de modelos de detecção.
Decide qual implementação usar baseado na disponibilidade do OpenVINO.
"""

import logging
from typing import Optional
from pathlib import Path

from src.domain.services.model_interface import IDetectionModel


logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory responsável por criar instâncias de modelos de detecção.
    Detecta automaticamente a disponibilidade do OpenVINO e seleciona
    a melhor implementação disponível.
    """
    
    @staticmethod
    def is_openvino_available() -> bool:
        """
        Verifica se o OpenVINO está disponível no sistema.
        
        :return: True se OpenVINO está disponível, False caso contrário.
        """
        try:
            import openvino
            logger.info(f"OpenVINO detectado: versão {openvino.__version__}")
            return True
        except ImportError:
            logger.info("OpenVINO não está disponível. Usando implementação padrão.")
            return False
    
    @staticmethod
    def create_model(
        model_path: str,
        use_openvino: bool = True,
        openvino_device: str = "AUTO",
        openvino_precision: str = "FP16"
    ) -> IDetectionModel:
        """
        Cria uma instância de modelo de detecção.
        
        :param model_path: Caminho para o arquivo do modelo.
        :param use_openvino: Se deve tentar usar OpenVINO (padrão: True).
        :param openvino_device: Dispositivo OpenVINO (AUTO, CPU, GPU, etc).
        :param openvino_precision: Precisão do modelo OpenVINO (FP16, FP32, INT8).
        :return: Instância de IDetectionModel.
        """
        from src.infrastructure.model.yolo_model_adapter import YOLOModelAdapter
        from src.infrastructure.model.openvino_model_adapter import OpenVINOModelAdapter
        
        model_path_obj = Path(model_path)
        
        # Verifica se deve usar OpenVINO
        if use_openvino and ModelFactory.is_openvino_available():
            try:
                logger.info(
                    f"Tentando carregar modelo com OpenVINO "
                    f"(device={openvino_device}, precision={openvino_precision})"
                )
                return OpenVINOModelAdapter(
                    model_path=str(model_path_obj),
                    device=openvino_device,
                    precision=openvino_precision
                )
            except Exception as e:
                logger.warning(
                    f"Falha ao carregar modelo com OpenVINO: {e}. "
                    "Fallback para implementação padrão YOLO."
                )
        
        # Fallback: usa implementação padrão YOLO
        logger.info("Carregando modelo com implementação padrão YOLO")
        return YOLOModelAdapter(model_path=str(model_path_obj))