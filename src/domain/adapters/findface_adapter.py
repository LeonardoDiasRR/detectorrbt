"""
Adapter para integração com FindFace Multi API.
Implementa o padrão Adapter do DDD para isolar o domínio da infraestrutura externa.
"""

# built-in
from typing import List, Dict, Optional
import logging
import json

# local
from findface_multi import FindfaceMulti
from src.domain.entities import Camera, Event
from src.domain.value_objects import CameraTokenVO, IdVO, NameVO, CameraSourceVO


class FindfaceAdapter:
    """
    Adapter que encapsula a comunicação com a API FindFace Multi.
    Traduz entre objetos de domínio e a API externa, protegendo o domínio
    de mudanças na infraestrutura externa.
    """

    def __init__(self, findface: FindfaceMulti, camera_prefix: str = 'EXTERNO'):
        """
        Inicializa o adapter do FindFace.

        :param findface: Instância do cliente FindfaceMulti.
        :param camera_prefix: Prefixo para filtrar câmeras virtuais.
        :raises TypeError: Se findface não for FindfaceMulti.
        """
        if not isinstance(findface, FindfaceMulti):
            raise TypeError("O parâmetro 'findface' deve ser uma instância de FindfaceMulti.")
        
        self.findface = findface
        self.camera_prefix = camera_prefix
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_cameras(self) -> List[Camera]:
        """
        Obtém a lista de câmeras virtuais disponíveis no FindFace.
        Retorna entidades Camera do domínio.

        :return: Lista de entidades Camera.
        """
        cameras = []

        try:
            # Obtém grupos de câmeras
            grupos = self.findface.get_camera_groups()["results"]
            grupos_filtrados = [
                g for g in grupos 
                if g["name"].lower().startswith(self.camera_prefix.lower())
            ]

            # Para cada grupo, obtém câmeras
            for grupo in grupos_filtrados:
                cameras_response = self.findface.get_cameras(
                    camera_groups=[grupo["id"]],
                    external_detector=True,
                    ordering='id'
                )["results"]
                
                # Filtra câmeras com RTSP no comment
                cameras_filtradas = [
                    c for c in cameras_response 
                    if c["comment"].startswith("rtsp://")
                ]
                
                # Converte para entidades Camera
                for camera_data in cameras_filtradas:
                    camera = Camera(
                        camera_id=IdVO(camera_data["id"]),
                        camera_name=NameVO(camera_data["name"]),
                        camera_token=CameraTokenVO(camera_data["external_detector_token"]),
                        source=CameraSourceVO(camera_data["comment"].strip())
                    )
                    cameras.append(camera)

            self.logger.info(f"Obtidas {len(cameras)} câmeras do FindFace")
            return cameras

        except Exception as e:
            self.logger.error(f"Erro ao obter câmeras do FindFace: {e}", exc_info=True)
            return []

    def send_event(self, event: Event) -> Optional[Dict]:
        """
        Envia um evento de face para o FindFace.
        Converte a entidade Event do domínio para o formato esperado pela API.

        :param event: Entidade Event do domínio.
        :return: Resposta do FindFace ou None em caso de erro.
        :raises TypeError: Se event não for Event.
        """
        if not isinstance(event, Event):
            raise TypeError(f"event deve ser Event, recebido: {type(event).__name__}")

        try:
            # Converte frame para JPEG
            imagem_bytes = event.frame.jpg(quality=95)
            
            # Expande bbox em 20% na diagonal
            x1, y1, x2, y2 = event.bbox.value()
            width = x2 - x1
            height = y2 - y1
            
            # Calcula expansão de 20% em cada direção
            expand_w = width * 0.20
            expand_h = height * 0.20
            
            # Aplica expansão mantendo dentro dos limites do frame
            frame_height, frame_width = event.frame.ndarray.shape[:2]
            x1_expanded = max(0, x1 - expand_w)
            y1_expanded = max(0, y1 - expand_h)
            x2_expanded = min(frame_width, x2 + expand_w)
            y2_expanded = min(frame_height, y2 + expand_h)
            
            # Converte bbox expandido para formato ROI [left, top, right, bottom]
            roi = [int(x1_expanded), int(y1_expanded), int(x2_expanded), int(y2_expanded)]
            
            # Converte timestamp para formato ISO 8601 com timezone local
            timestamp_iso = event.frame.timestamp.value().astimezone().isoformat()
            
            # Envia para FindFace
            resposta = self.findface.add_face_event(
                token=event.camera_token.value(),
                fullframe=imagem_bytes,
                camera=event.camera_id.value(),
                roi=roi,
                mf_selector="all",
                timestamp=timestamp_iso
            )
            
            self.logger.debug(
                f"Evento enviado para FindFace - Camera: {event.camera_id.value()}, "
                f"Quality: {event.face_quality_score.value():.4f}, "
                f"Timestamp: {timestamp_iso}"
            )
            
            return resposta

        except Exception as e:
            # Tenta extrair a mensagem 'desc' do payload retornado pelo servidor FindFace
            desc = None
            try:
                # Se a exceção for um dicionário já
                if isinstance(e, dict):
                    desc = e.get("desc")
                else:
                    # Se a exception tiver .response (requests/http errors), tenta JSON
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        try:
                            data = resp.json()
                        except Exception:
                            try:
                                data = json.loads(getattr(resp, 'text', '') or '')
                            except Exception:
                                data = None
                        if isinstance(data, dict):
                            desc = data.get("desc")

                    # Se ainda sem desc, tenta parsear a string da exceção como JSON
                    if desc is None:
                        try:
                            parsed = json.loads(str(e))
                            if isinstance(parsed, dict):
                                desc = parsed.get("desc")
                        except Exception:
                            desc = None
            except Exception:
                desc = None

            if desc:
                # Loga apenas o campo 'desc' conforme solicitado
                self.logger.error(f"Erro ao enviar evento para FindFace - Camera: {event.camera_id.value()}: {desc}")
            else:
                # Fallback: log completo com stacktrace para investigação
                self.logger.error(
                    f"Erro ao enviar evento para FindFace - Camera: {event.camera_id.value()}: {e}",
                    exc_info=True
                )
            return None