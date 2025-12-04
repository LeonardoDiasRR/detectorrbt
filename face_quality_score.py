import numpy as np
import math
from typing import Tuple, Optional
import os
from pathlib import Path
import yaml
import cv2


# Carrega configuração (qualidade_face) a partir do arquivo config.yaml no mesmo diretório
_DEFAULT_QUALIDADE = {"tamanho_bbox": 2, "face_frontal": 6}
qualidade_cfg = {}
try:
    _cfg_path = Path(__file__).parent / "config.yaml"
    if _cfg_path.exists():
        with open(_cfg_path, "r", encoding="utf-8") as _f:
            _cfg = yaml.safe_load(_f) or {}
            if isinstance(_cfg, dict):
                _qc = _cfg.get("qualidade_face")
                if isinstance(_qc, dict):
                    # coerce numeric values
                    qualidade_cfg = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in _qc.items()}
except Exception:
    qualidade_cfg = {}

# garante valores padrão
for _k, _v in _DEFAULT_QUALIDADE.items():
    qualidade_cfg.setdefault(_k, _v)

def get_sharpness_score(imagem: np.ndarray) -> float:
    """
    Calcula uma medida normalizada de nitidez baseada na variância do Laplaciano.

    Parâmetros:
        imagem (np.ndarray): Imagem em escala de cinza ou colorida (será convertida).

    Retorno:
        float: Valor entre 0 e 1 (quanto maior, mais nítida a imagem).
               Retorna 0.01 se a imagem for inválida ou vazia.
    """
    # Verificação de tipo
    if not isinstance(imagem, np.ndarray):
        raise TypeError("imagem deve ser um objeto numpy.ndarray.")

    # Converte para escala de cinza se necessário
    if len(imagem.shape) == 3:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Verifica se a imagem é válida
    if imagem is None or imagem.size == 0:
        return 0.01

    # Calcula a variância do filtro Laplaciano
    laplacian_var = float(cv2.Laplacian(imagem, cv2.CV_64F).var())

    # Normaliza a variância para faixa [0, 1]
    valor_maximo_esperado = 1000.0
    nitidez_normalizada = min(laplacian_var / valor_maximo_esperado, 1.0)

    return nitidez_normalizada if nitidez_normalizada > 0 else 0.01

def get_face_bbox_score(bbox: Tuple[int, int, int, int]) -> float:
    """
    Calcula um score de proporção altura/largura da bounding box,
    onde valores próximos da razão típica de um rosto (1.2) retornam score próximo de 1.0.

    Parâmetros:
        bbox (Tuple[int, int, int, int]): Coordenadas da bounding box (x1, y1, x2, y2)

    Retorna:
        float: Score entre 0 e 1
    """
    if not isinstance(bbox, tuple) or len(bbox) != 4:
        raise ValueError("bbox deve ser uma tupla com 4 coordenadas")

    x1, y1, x2, y2 = map(int, bbox)
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    proporcao = height / width

    proporcao_ideal = 1.2  # Proporção típica de face (altura/largura)
    erro = abs(proporcao - proporcao_ideal)
    
    # Score decresce conforme se afasta da proporção ideal
    score = 1.0 - min(erro / proporcao_ideal, 1.0)
    return round(score, 3)


def get_face_score(landmarks: np.ndarray) -> float:
    """
    Calcula um índice de frontalidade facial com base na simetria dos landmarks.

    Parâmetros:
        landmarks (np.ndarray): Array numpy de shape (5, 2) contendo os 5 pontos da face:
                                [olho esquerdo, olho direito, nariz, canto esquerdo da boca, canto direito da boca].

    Retorno:
        float: Score de frontalidade entre 0.0 e 1.0. Quanto mais próximo de 1.0, mais frontal a face.

    Exceções:
        TypeError: Se landmarks não for um np.ndarray.
        ValueError: Se landmarks não tiver shape (5, 2).
    """
    # Verificação de tipo
    if not isinstance(landmarks, np.ndarray):
        raise TypeError("landmarks deve ser um np.ndarray.")
    
    # Verificação de shape
    if landmarks.shape != (5, 2):
        raise ValueError("landmarks deve ter shape (5, 2).")

    # Desempacota os pontos: olho esq., olho dir., nariz, boca esq., boca dir.
    le, re, n, lm, rm = landmarks

    # Calcula as distâncias entre o nariz e os demais pontos
    dist_n_le = math.dist(n, le)
    dist_n_re = math.dist(n, re)
    dist_n_lm = math.dist(n, lm)
    dist_n_rm = math.dist(n, rm)

    # Distância média usada para normalização
    avg_dist = (dist_n_le + dist_n_re + dist_n_lm + dist_n_rm) / 4.0

    # Diferença de simetria entre olhos e entre cantos da boca
    symmetry_diff = abs(dist_n_le - dist_n_re) + abs(dist_n_lm - dist_n_rm)

    # Evita divisão por zero
    epsilon = 1e-6

    # Score de frontalidade (quanto mais simétrico, mais próximo de 1.0)
    score = 1.0 - (symmetry_diff / (2.0 * avg_dist + epsilon))

    # log_track.info(f'Face Score: {max(0.0, min(1.0, score))}')

    # Garante que o score esteja dentro do intervalo [0.0, 1.0]
    return max(0.0, min(1.0, score))


def get_face_quality_score(
    bbox: Tuple[int, int, int, int],
    confidence: float,
    frame: np.ndarray,
    landmarks: Optional[np.ndarray] = None
) -> float:
    """
    Calcula um índice de qualidade da face baseado em múltiplas heurísticas:
    - Confiança da detecção (YOLO) (peso configurável)
    - Tamanho relativo da bounding box no frame (peso configurável)
    - Índice de frontalidade com base nos landmarks (peso configurável)

    Parâmetros:
        bbox (Tuple[int, int, int, int]): Bounding box no formato (x1, y1, x2, y2)
        confidence (float): Confiança da detecção YOLO (0 a 1)
        frame (np.ndarray): Frame original contendo a imagem
        landmarks (np.ndarray, opcional): Array de shape (5, 2) com landmarks faciais ou None

    Retorna:
        float: Score de qualidade da face (entre 0.0 e 1.0)
    """
    # Verificações de tipo
    if not isinstance(bbox, tuple):
        raise TypeError("bbox deve ser uma tupla.")
    if not isinstance(confidence, float):
        raise TypeError("confidence deve ser float.")
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame deve ser np.ndarray.")
    if landmarks is not None and not isinstance(landmarks, np.ndarray):
        raise TypeError("landmarks deve ser um np.ndarray ou None.")

    # Extrair dimensões do frame
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    # Coordenadas da bbox
    x1, y1, x2, y2 = map(int, bbox)
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    bbox_area = w * h

    # Tamanho relativo da bbox (limitado a 10% da área do frame)
    area_ratio = min(bbox_area / frame_area, 0.1) / 0.1

    # Cálculo da frontalidade
    if landmarks is not None:
        frontal_ratio = get_face_score(landmarks)
    else:
        frontal_ratio = 0.01


    # Pesos configuráveis (lidos de config.yaml em tempo de importação)
    # `qualidade_cfg` é carregado no escopo do módulo a partir do arquivo `config.yaml`.
    peso_tamanho_bbox = qualidade_cfg.get("tamanho_bbox", 2)
    peso_face_frontal = qualidade_cfg.get("face_frontal", 6)


    # Score final ponderado
    score = (
        (area_ratio * peso_tamanho_bbox) +
        (frontal_ratio * peso_face_frontal)
    )

    peso_total = (
        peso_tamanho_bbox +
        peso_face_frontal
    )

    return np.clip(score / peso_total, 0.0, 1.0)