"""
Entidade Frame do domínio.
"""

from typing import Optional, Tuple
import numpy as np
import cv2
from domain.value_objects import IdVO, NameVO, CameraTokenVO, TimestampVO


class Frame:
    """
    Entidade que representa um frame capturado de uma câmera.
    """

    def __init__(
        self,
        id: IdVO,
        ndarray: np.ndarray,
        camera_id: IdVO,
        camera_name: NameVO,
        camera_token: CameraTokenVO,
        timestamp: TimestampVO
    ):
        """
        Inicializa a entidade Frame.

        :param id: ID único do frame (IdVO).
        :param ndarray: Array numpy representando a imagem do frame.
        :param camera_id: ID da câmera que capturou o frame (IdVO).
        :param camera_name: Nome da câmera que capturou o frame (NameVO).
        :param camera_token: Token de autenticação da câmera (CameraTokenVO).
        :param timestamp: Timestamp de captura do frame (TimestampVO).
        :raises TypeError: Se algum parâmetro não for do tipo esperado.
        :raises ValueError: Se o ndarray for inválido.
        """
        if not isinstance(id, IdVO):
            raise TypeError(f"id deve ser IdVO, recebido: {type(id).__name__}")
        
        if not isinstance(ndarray, np.ndarray):
            raise TypeError(f"ndarray deve ser np.ndarray, recebido: {type(ndarray).__name__}")
        
        if ndarray.size == 0:
            raise ValueError("ndarray não pode ser vazio")
        
        if not isinstance(camera_id, IdVO):
            raise TypeError(f"camera_id deve ser IdVO, recebido: {type(camera_id).__name__}")
        
        if not isinstance(camera_name, NameVO):
            raise TypeError(f"camera_name deve ser NameVO, recebido: {type(camera_name).__name__}")
        
        if not isinstance(camera_token, CameraTokenVO):
            raise TypeError(f"camera_token deve ser CameraTokenVO, recebido: {type(camera_token).__name__}")
        
        if not isinstance(timestamp, TimestampVO):
            raise TypeError(f"timestamp deve ser TimestampVO, recebido: {type(timestamp).__name__}")
        
        self._id = id
        self._ndarray = ndarray
        self._camera_id = camera_id
        self._camera_name = camera_name
        self._camera_token = camera_token
        self._timestamp = timestamp

    @property
    def id(self) -> IdVO:
        """Retorna o ID do frame."""
        return self._id

    @property
    def ndarray(self) -> np.ndarray:
        """Retorna o array numpy do frame."""
        return self._ndarray

    @property
    def camera_id(self) -> IdVO:
        """Retorna o ID da câmera."""
        return self._camera_id

    @property
    def camera_name(self) -> NameVO:
        """Retorna o nome da câmera."""
        return self._camera_name

    @property
    def camera_token(self) -> CameraTokenVO:
        """Retorna o token da câmera."""
        return self._camera_token

    @property
    def timestamp(self) -> TimestampVO:
        """Retorna o timestamp de captura do frame."""
        return self._timestamp

    def jpg(self, quality: int = 95) -> bytes:
        """
        Converte o frame para formato JPEG e retorna como bytes.

        :param quality: Qualidade de compressão JPEG (0-100), padrão 95.
        :return: Frame codificado em JPEG como bytes.
        :raises ValueError: Se a qualidade for inválida.
        :raises RuntimeError: Se a codificação falhar.
        """
        if not 0 <= quality <= 100:
            raise ValueError(f"Qualidade deve estar entre 0 e 100, recebido: {quality}")
        
        success, buffer = cv2.imencode('.jpg', self._ndarray, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        if not success:
            raise RuntimeError("Falha ao codificar o frame em JPEG")
        
        return buffer.tobytes()

    @property
    def shape(self) -> Tuple[int, ...]:
        """Retorna as dimensões do frame (altura, largura, canais)."""
        return self._ndarray.shape

    @property
    def height(self) -> int:
        """Retorna a altura do frame."""
        return self._ndarray.shape[0]

    @property
    def width(self) -> int:
        """Retorna a largura do frame."""
        return self._ndarray.shape[1]

    def copy(self) -> 'Frame':
        """
        Cria uma cópia do frame.

        :return: Nova instância de Frame com ndarray copiado.
        """
        return Frame(
            id=self._id,
            ndarray=self._ndarray.copy(),
            camera_id=self._camera_id,
            camera_name=self._camera_name,
            camera_token=self._camera_token,
            timestamp=self._timestamp
        )

    def __eq__(self, other) -> bool:
        """Compara dois frames por igualdade (baseado no ID)."""
        if not isinstance(other, Frame):
            return False
        return self._id == other._id

    def __hash__(self) -> int:
        """Retorna o hash do frame (baseado no ID)."""
        return hash(self._id)

    def __repr__(self) -> str:
        """Representação string do frame."""
        return (
            f"Frame(id={self._id.value()}, "
            f"shape={self.shape}, "
            f"camera_id={self._camera_id.value()}, "
            f"camera_name='{self._camera_name.value()}')"
        )

    def __str__(self) -> str:
        """Conversão para string."""
        return f"Frame {self._id.value()} from Camera {self._camera_id.value()}: {self.width}x{self.height}"
