# built-in
from typing import Optional, Tuple
import math

# 3rd party
import numpy as np
import cv2

# local
from config_loader import CONFIG


def get_face_quality_score(
    bbox: Tuple[int, int, int, int],
    confidence: float,
    frame: np.ndarray,
    landmarks: Optional[np.ndarray] = None
) -> float:
    """
    Calcula um score de qualidade para uma face detectada.
    
    :param bbox: Bounding box (x1, y1, x2, y2).
    :param confidence: Confiança da detecção YOLO.
    :param frame: Frame original.
    :param landmarks: Landmarks da face (opcional).
    :return: Score de qualidade (0.0 a 1.0).
    """
    
    # Obtém pesos das configurações
    peso_confianca = CONFIG.get("qualidade_face", {}).get("confianca_deteccao", 3)
    peso_tamanho = CONFIG.get("qualidade_face", {}).get("tamanho_bbox", 4)
    peso_frontal = CONFIG.get("qualidade_face", {}).get("face_frontal", 6)
    peso_proporcao = CONFIG.get("qualidade_face", {}).get("proporcao_bbox", 1)
    peso_nitidez = CONFIG.get("qualidade_face", {}).get("nitidez", 1)
    
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    # 1. Score de confiança (já normalizado 0-1)
    score_confianca = confidence
    
    # 2. Score de tamanho (normalizado pela área do frame)
    frame_area = frame.shape[0] * frame.shape[1]
    score_tamanho = min(area / (frame_area * 0.3), 1.0)  # Máximo em 30% do frame
    
    # 3. Score de face frontal (baseado em landmarks se disponível)
    score_frontal = 1.0
    if landmarks is not None and len(landmarks) >= 5:
        # Calcula simetria dos olhos
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        
        # Distância horizontal dos olhos
        eye_distance = abs(right_eye[0] - left_eye[0])
        
        # Distância do nariz ao centro dos olhos
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        nose_offset = abs(nose[0] - eye_center_x)
        
        # Score frontal: quanto menor o offset do nariz, mais frontal
        if eye_distance > 0:
            score_frontal = max(0.0, 1.0 - (nose_offset / eye_distance))
    
    # 4. Score de proporção (faces próximas de 1:1.3 são ideais)
    aspect_ratio = height / width if width > 0 else 0
    ideal_ratio = 1.3
    ratio_diff = abs(aspect_ratio - ideal_ratio)
    score_proporcao = max(0.0, 1.0 - ratio_diff)
    
    # 5. Score de nitidez (variação Laplaciana)
    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size > 0:
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        # Normaliza (valores típicos: 0-1000)
        score_nitidez = min(laplacian_var / 500.0, 1.0)
    else:
        score_nitidez = 0.0
    
    # Calcula score final ponderado
    total_peso = peso_confianca + peso_tamanho + peso_frontal + peso_proporcao + peso_nitidez
    
    score_final = (
        score_confianca * peso_confianca +
        score_tamanho * peso_tamanho +
        score_frontal * peso_frontal +
        score_proporcao * peso_proporcao +
        score_nitidez * peso_nitidez
    ) / total_peso
    
    return score_final
