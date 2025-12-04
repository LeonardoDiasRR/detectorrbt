# built-in
import cv2
import numpy as np
from typing import Optional, Tuple, Any, List, Union
from datetime import datetime

# 3rd-party
from frame import Frame

class Event:
    """
    Representa um evento de detecção facial em um frame específico.

    Atributos:
        id (int): Identificador único do evento.
        frame (Frame): Objeto Frame associado.
        bbox (Tuple[int, int, int, int]): Coordenadas da bounding box (x1, y1, x2, y2).
        confianca (float): Nível de confiança da detecção.
        landmarks (Optional[np.ndarray]): Landmarks faciais (5x2), se disponíveis.
        track (Optional[Track]): Track associado ao evento.
        thumbnail (np.ndarray): Crop da face com 20% de margem.
    """

    _id_counter = 0

    def __init__(
        self,
        frame: Frame,
        bbox: Tuple[int, int, int, int],
        bbox_frame_id: str,
        confianca: float,
        landmarks: Optional[np.ndarray] = None
    ) -> None:
        # Validação dos tipos de entrada
        if not isinstance(frame, Frame):
            raise TypeError("frame deve ser uma instância da classe Frame.")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise TypeError("bbox deve ser uma tupla com 4 inteiros.")
        if not isinstance(bbox_frame_id, str):
            raise TypeError("bbox_frame_id deve ser uma string.")
        if not isinstance(confianca, float):
            raise TypeError("confianca deve ser float.")
        if landmarks is not None and not isinstance(landmarks, np.ndarray):
            raise TypeError("landmarks deve ser um np.ndarray ou None.")

        Event._id_counter += 1
        self.id: int = Event._id_counter
        self.frame: Frame = frame
        self.bbox: Tuple[int, int, int, int] = bbox
        self.bbox_frame_id: str = bbox_frame_id
        self.confianca: float = confianca
        self.landmarks: Optional[np.ndarray] = landmarks
        self._timestamp: datetime = datetime.now()
        self.track: Optional[Track] = None

    @property
    def largura(self) -> int:
        """Largura da bounding box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def altura(self) -> int:
        """Altura da bounding box."""
        return self.bbox[3] - self.bbox[1]

    @property
    def centro(self) -> Tuple[int, int]:
        """Centro (x, y) da bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def timestamp(self) -> datetime:
        """Horário de criação do evento."""
        return self._timestamp

    @property
    def face_quality(self) -> float:
        """
        Score de qualidade da face.
        Calculado com base em múltiplas heurísticas via função utilitária.
        """
        return get_face_quality_score(
            bbox=self.bbox,
            confidence=self.confianca,
            frame=self.frame.ndarray,
            landmarks=self.landmarks
        )

    @property
    def thumbnail(self) -> np.ndarray:
        """
        Gera a thumbnail da face com 20% de margem em torno da bounding box.

        Retorna:
            np.ndarray: Imagem da face recortada.
        """
        return gerar_thumbnail(self.frame.ndarray, self.bbox)

    @property
    def frame_formatado(self) -> np.ndarray:
        """Retorna o frame formatado conforme configuração para envio ao FindFace.
        
        Formatos disponíveis:
        - "blur": Frame completo com blur nas outras faces (sem rodapé)
        - "body_crop": Crop expandido do corpo com rodapé

        Retorna:
            np.ndarray: Frame formatado pronto para envio.
        """
        formato = CONFIG.get("formato_exportacao_findface", "blur")
        
        # Validação do formato
        if formato not in ["blur", "body_crop"]:
            log_evento.warning(
                f"Formato inválido '{formato}' em formato_exportacao_findface. "
                f"Usando 'blur' como padrão."
            )
            formato = "blur"
        
        if formato == "body_crop":
            return self._body_crop_original()
        else:
            return self._aplicar_blur_frame()
    
    def _aplicar_blur_frame(self) -> np.ndarray:
        """Aplica blur nas outras faces do frame (modo blur, sem rodapé).
        
        Retorna:
            np.ndarray: Frame completo com blur nas outras faces.
        """
        # Filtra todos os bboxes EXCETO o deste evento (usa cache!)
        outras_faces_coords = [
            bbox.value() 
            for bbox in self.frame.bboxes_detectados 
            if bbox.id != self.bbox_frame_id
        ]
        
        # Aplica blur nas outras faces
        frame_com_blur = aplicar_blur_faces(
            frame=self.frame.ndarray,
            bbox_evento=self.bbox,
            outras_bboxes=outras_faces_coords,
            blur_kernel_size=CONFIG.get("blur_kernel_size", 99),
            blur_margem=CONFIG.get("blur_margem_expansao", 0.1),
            blur_cor=hex_para_rgb(CONFIG.get("blur_cor", "#A0A0A0")),
            blur_intensidade=CONFIG.get("blur_intensidade", 0.7)
        )
        
        return frame_com_blur
    
    def _body_crop_original(self) -> np.ndarray:
        """Implementação do crop expandido do corpo (modo body_crop, com rodapé).
        
        Retorna:
            np.ndarray: Crop do corpo expandido com rodapé.
        """
        x1, y1, x2, y2 = self.bbox
        largura = x2 - x1
        altura = y2 - y1

        # Cálculo do bounding box expandido
        x_crop = int(x1 - 1.25 * largura)
        y_crop = int(y1 - 0.2 * altura)
        x2_crop = int(x1 + 2.5 * largura)
        y2_crop = int(y1 + 7.0 * altura)

        frame_altura, frame_largura = self.frame.ndarray.shape[:2]

        # Garantindo que o bounding box esteja dentro dos limites do frame
        x_crop = max(0, x_crop)
        y_crop = max(0, y_crop)
        x2_crop = min(frame_largura, x2_crop)
        y2_crop = min(frame_altura, y2_crop)

        crop = self.frame.ndarray[y_crop:y2_crop, x_crop:x2_crop].copy()
        return self._adicionar_rodape(crop)
    
    def _adicionar_rodape(self, frame: np.ndarray) -> np.ndarray:
        """Adiciona rodapé branco no frame com nome da câmera e timestamp.
        
        Parâmetros:
            frame: Frame ao qual adicionar o rodapé
            
        Retorna:
            Frame com rodapé adicionado (10% da altura)
        """
        timestamp: str = self.frame.timestamp.strftime(r"%Y-%m-%d %H:%M:%S")
        camera_name: str = self.frame.camera_name or ""
        frame_altura, frame_largura = frame.shape[:2]
        barra_altura: int = int(frame_altura * 0.10)
        
        # Cria frame expandido com espaço para rodapé
        frame_com_rodape = np.vstack([
            frame,
            np.full((barra_altura, frame_largura, 3), 255, dtype=np.uint8)
        ])
        
        y_inicio = frame_altura
        espaco: int = int(barra_altura * 0.05)
        linha_altura: int = (barra_altura - espaco) // 2
        
        # Texto da câmera
        base_cam = cv2.getTextSize(camera_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
        escala_largura_cam = (frame_largura * 0.95) / base_cam[0] if base_cam[0] else 1
        escala_altura_cam = (linha_altura * 0.8) / base_cam[1] if base_cam[1] else 1
        font_cam = min(escala_largura_cam, escala_altura_cam)
        thick_cam = max(1, int(font_cam * 2))
        tamanho_cam = cv2.getTextSize(camera_name, cv2.FONT_HERSHEY_SIMPLEX, font_cam, thick_cam)[0]
        x_cam = (frame_largura - tamanho_cam[0]) // 2
        y_cam = y_inicio + (linha_altura + tamanho_cam[1]) // 2
        
        cv2.putText(frame_com_rodape, camera_name, (x_cam, y_cam),
                    cv2.FONT_HERSHEY_SIMPLEX, font_cam, (0, 0, 0), thick_cam, cv2.LINE_AA)
        
        # Texto do timestamp
        base_ts = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
        escala_largura_ts = (frame_largura * 0.95) / base_ts[0]
        escala_altura_ts = (linha_altura * 0.8) / base_ts[1]
        font_ts = min(escala_largura_ts, escala_altura_ts)
        thick_ts = max(1, int(font_ts * 2))
        tamanho_ts = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, font_ts, thick_ts)[0]
        x_ts = (frame_largura - tamanho_ts[0]) // 2
        y_ts = y_inicio + linha_altura + espaco + (linha_altura + tamanho_ts[1]) // 2
        
        cv2.putText(frame_com_rodape, timestamp, (x_ts, y_ts),
                    cv2.FONT_HERSHEY_SIMPLEX, font_ts, (0, 0, 0), thick_ts, cv2.LINE_AA)
        
        return frame_com_rodape
