# built-in
from typing import Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import logging

# 3rd party
import numpy as np
import cv2

# local
from findface_adapter import enviar_imagem_para_findface
from findface_multi import FindfaceMulti


class TrackProcessor:
    """
    Classe responsável por processar tracks finalizados do ByteTrackDetector.
    Desacopla o processamento de cada track, incluindo seleção da melhor face,
    salvamento em disco e envio para o FindFace.
    """

    def __init__(
        self,
        track_id: int,
        detections: List[Tuple[np.ndarray, float, datetime, Tuple[int, int, int, int], Optional[np.ndarray], float]],
        camera_id: int,
        camera_name: str,
        camera_token: str,
        findface: Optional[FindfaceMulti] = None,
        project: str = "./imagens/",
        name: str = "rtsp_byte_track_results"
    ):
        """
        Inicializa o processador de track.

        :param track_id: ID do track a ser processado.
        :param detections: Lista de detecções do track [(frame, score, timestamp, bbox, landmarks, conf), ...].
        :param camera_id: ID da câmera.
        :param camera_name: Nome da câmera.
        :param camera_token: Token da câmera para envio ao FindFace.
        :param findface: Instância do FindfaceMulti para envio de eventos (opcional).
        :param project: Diretório base para salvamento de imagens.
        :param name: Nome do subdiretório para salvamento.
        """
        self.track_id = track_id
        self.detections = detections
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.camera_token = camera_token
        self.findface = findface
        self.project = project
        self.name = name
        self.logger = logging.getLogger(f"TrackProcessor_{camera_id}_{camera_name}")

    def process(self) -> None:
        """
        Processa o track: encontra a melhor face, salva em disco e envia para o FindFace.
        """
        if not self.detections or len(self.detections) == 0:
            self.logger.warning(f"Track {self.track_id} não possui detecções para processar")
            return

        total_detections = len(self.detections)
        self.logger.info(f"Processando Track {self.track_id} com {total_detections} detecções")

        # Encontra o frame com melhor qualidade
        best_data = max(self.detections, key=lambda x: x[1])
        
        # Desempacota com confiança YOLO
        if len(best_data) >= 6:
            best_frame, best_score, best_timestamp, best_bbox, best_landmarks, best_conf = best_data
        else:
            best_frame, best_score, best_timestamp, best_bbox, best_landmarks = best_data
            best_conf = None

        # Salva a melhor face
        self._save_best_frame(best_frame, best_score, best_timestamp, best_bbox, total_detections, best_conf)

        # Envia para o FindFace
        if self.findface is not None:
            self._send_best_frame_to_findface(best_frame, best_score, best_timestamp, best_bbox, total_detections)

    def _save_best_frame(
        self,
        frame: np.ndarray,
        score: float,
        timestamp: datetime,
        bbox: Tuple[int, int, int, int],
        total_detections: int,
        conf: Optional[float] = None
    ) -> None:
        """
        Salva o frame com bbox desenhado.

        :param frame: Frame numpy array.
        :param score: Score de qualidade da face.
        :param timestamp: Timestamp da detecção.
        :param bbox: Bounding box (x1, y1, x2, y2).
        :param total_detections: Total de detecções no track.
        :param conf: Confiança YOLO da detecção (opcional).
        """
        try:
            x1, y1, x2, y2 = bbox

            # Desenha bbox verde
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label com confiança YOLO
            if conf is not None:
                label = f"Track {self.track_id} | Score: {score:.4f} | Conf: {conf:.2f}"
            else:
                label = f"Track {self.track_id} | Score: {score:.4f}"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

            # Nome do arquivo
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"Track_{self.track_id}_{timestamp_str}.jpg"
            filepath = Path(self.project) / self.name / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Salva
            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            conf_str = f", conf={conf:.2f}" if conf is not None else ""
            self.logger.info(
                f"Melhor face salva: {filename} (score={score:.4f}{conf_str}) | "
                f"Total de detecções: {total_detections}"
            )

        except Exception as e:
            self.logger.error(f"Erro ao salvar frame do track {self.track_id}: {e}", exc_info=True)

    def _send_best_frame_to_findface(
        self,
        frame: np.ndarray,
        score: float,
        timestamp: datetime,
        bbox: Tuple[int, int, int, int],
        total_detections: int
    ) -> None:
        """
        Envia o melhor frame do track para o FindFace.

        :param frame: Frame numpy array.
        :param score: Score de qualidade da face.
        :param timestamp: Timestamp da detecção.
        :param bbox: Bounding box (x1, y1, x2, y2).
        :param total_detections: Total de detecções no track.
        """
        try:
            # Converte o frame para bytes (JPEG)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            imagem_bytes = buffer.tobytes()

            # Envia para o FindFace
            resposta = enviar_imagem_para_findface(
                findface=self.findface,
                camera_id=self.camera_id,
                camera_token=self.camera_token,
                imagem_bytes=imagem_bytes,
                bbox=bbox
            )

            # Verifica se o envio foi bem-sucedido
            if resposta:
                self.logger.info(
                    f"✓ FindFace - Envio BEM-SUCEDIDO - Track {self.track_id}: "
                    f"score={score:.4f}, total_detections={total_detections}, "
                    f"camera_id={self.camera_id}, resposta={resposta}"
                )
            else:
                self.logger.warning(
                    f"✗ FindFace - Envio retornou resposta vazia - Track {self.track_id}: "
                    f"score={score:.4f}, total_detections={total_detections}, "
                    f"camera_id={self.camera_id}"
                )

        except Exception as e:
            self.logger.error(
                f"✗ FindFace - FALHA no envio - Track {self.track_id}: "
                f"score={score:.4f}, total_detections={total_detections}, "
                f"camera_id={self.camera_id}, erro={e}",
                exc_info=True
            )
