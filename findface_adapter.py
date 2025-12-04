# buitin
import os
import traceback
from dotenv import load_dotenv
import json

# 3d party
import yaml

# local
from findface_multi import FindfaceMulti

def obter_lista_cameras_virtuais_findface(findface: FindfaceMulti, prefixo='EXTERNO') -> list[dict]:
    """
    Obtém a lista de câmeras virtuais disponíveis no FindFace com base no prefixo configurado.
    """

    if not isinstance(findface, FindfaceMulti):
        raise TypeError("O parâmetro 'findface' deve ser uma instância de FindfaceMulti.")
    
    lista_cameras = []

    grupos = findface.get_camera_groups()["results"]
    grupos_filtrados = [g for g in grupos if g["name"].lower().startswith(prefixo.lower())]

    for grupo in grupos_filtrados:
        cameras = findface.get_cameras(camera_groups=[grupo["id"]],
                                        external_detector=True,
                                        ordering='id')["results"]
        cameras_filtradas = [c for c in cameras if c["comment"].startswith("rtsp://")]
        for camera in cameras_filtradas:
            lista_cameras.append({
                "name": camera["name"],
                "url": camera["comment"].strip(),
                "token": camera["external_detector_token"],
                "id": camera["id"]
            })

    return lista_cameras


def enviar_imagem_para_findface(findface: FindfaceMulti, camera_id: int, camera_token: str, imagem_bytes: bytes, bbox: tuple) -> dict:
    """
    Envia uma imagem para o FindFace usando o token da câmera virtual.
    Retorna a resposta do FindFace como um dicionário.
    """

    if not isinstance(findface, FindfaceMulti):
        raise TypeError("O parâmetro 'findface' deve ser uma instância de FindfaceMulti.")
    
    if not isinstance(camera_id, int):
        raise ValueError("O parâmetro 'camera_id' deve ser um inteiro.")

    if not isinstance(camera_token, str) or not camera_token:
        raise ValueError("O parâmetro 'camera_token' deve ser uma string não vazia.")
    
    if not isinstance(imagem_bytes, bytes) or not imagem_bytes:
        raise ValueError("O parâmetro 'imagem_bytes' deve ser um objeto bytes não vazio.")
    
    if not isinstance(bbox, tuple):
        raise ValueError("O parâmetro 'bbox' deve ser uma tupla.")

    # Converte bbox (x1, y1, x2, y2) para o formato ROI esperado [left, top, right, bottom]
    roi = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] if len(bbox) >= 4 else None

    resposta = findface.add_face_event(
        token=camera_token,
        fullframe=imagem_bytes,
        camera=camera_id,
        roi=roi,
        mf_selector="biggest"
    )
    
    return resposta

