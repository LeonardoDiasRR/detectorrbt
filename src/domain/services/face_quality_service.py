"""
Serviço de domínio para cálculo de qualidade facial.
"""

from typing import Optional
import numpy as np
import cv2

from src.domain.value_objects import BboxVO, ConfidenceVO, LandmarksVO
from src.domain.entities.frame_entity import Frame


class FaceQualityService:
    """
    Serviço de domínio responsável por calcular a qualidade de uma face detectada.
    Utiliza múltiplos critérios ponderados para determinar um score de qualidade.
    """

    @staticmethod
    def _calculate_confidence_score(confidence: ConfidenceVO) -> float:
        """
        Calcula o score baseado na confiança da detecção YOLO.

        :param confidence: Confiança da detecção YOLO.
        :return: Score de confiança (0.0 a 1.0).
        """
        return confidence.value()

    @staticmethod
    def _calculate_size_score(bbox: BboxVO, frame: Frame) -> float:
        """
        Calcula o score baseado no tamanho do bbox em relação ao frame.

        :param bbox: Bounding box da face.
        :param frame: Frame onde a face foi detectada.
        :return: Score de tamanho (0.0 a 1.0).
        """
        area = bbox.area
        frame_area = frame.height * frame.width
        # Máximo em 30% do frame
        return min(area / (frame_area * 0.3), 1.0)

    @staticmethod
    def _calculate_frontal_score(landmarks: LandmarksVO) -> float:
        """
        Calcula o score baseado na frontalidade da face usando landmarks.

        :param landmarks: Landmarks faciais.
        :return: Score de frontalidade (0.0 a 1.0).
        """
        landmarks_array = landmarks.value() if not landmarks.is_empty() else None
        
        if landmarks_array is None or len(landmarks_array) < 5:
            return 1.0
        
        # Calcula simetria dos olhos
        left_eye = landmarks_array[0]
        right_eye = landmarks_array[1]
        nose = landmarks_array[2]

        # Distância horizontal dos olhos
        eye_distance = abs(right_eye[0] - left_eye[0])

        # Distância do nariz ao centro dos olhos
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        nose_offset = abs(nose[0] - eye_center_x)

        # Score frontal: quanto menor o offset do nariz, mais frontal
        if eye_distance > 0:
            return max(0.0, 1.0 - (nose_offset / eye_distance))
        
        return 1.0

    @staticmethod
    def _calculate_proportion_score(bbox: BboxVO) -> float:
        """
        Calcula o score baseado na proporção do bbox.
        Faces próximas de 1:1.3 (altura:largura) são ideais.

        :param bbox: Bounding box da face.
        :return: Score de proporção (0.0 a 1.0).
        """
        width = bbox.width
        height = bbox.height
        
        if width == 0:
            return 0.0
        
        aspect_ratio = height / width
        ideal_ratio = 1.3
        ratio_diff = abs(aspect_ratio - ideal_ratio)
        return max(0.0, 1.0 - ratio_diff)

    @staticmethod
    def _calculate_sharpness_score(frame: Frame, bbox: BboxVO) -> float:
        """
        Calcula o score baseado na nitidez da face usando variação Laplaciana.
        OTIMIZAÇÃO: Usa ndarray_readonly para evitar cópia.

        :param frame: Frame onde a face foi detectada.
        :param bbox: Bounding box da face.
        :return: Score de nitidez (0.0 a 1.0).
        """
        x1, y1, x2, y2 = bbox.value()
        # OTIMIZAÇÃO: Usa ndarray_readonly - não precisa de cópia para leitura
        frame_ndarray = frame.ndarray_readonly
        
        face_roi = frame_ndarray[y1:y2, x1:x2].copy()  # Copia apenas o ROI pequeno
        
        if face_roi.size == 0:
            return 0.0
        
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        # Normaliza (valores típicos: 0-1000)
        return min(laplacian_var / 500.0, 1.0)

    @staticmethod
    def calculate_quality(
        frame: Frame,
        bbox: BboxVO,
        confidence: ConfidenceVO,
        landmarks: LandmarksVO,
        peso_confianca: float = 3,
        peso_tamanho: float = 4,
        peso_frontal: float = 6,
        peso_proporcao: float = 1,
        peso_nitidez: float = 1
    ) -> ConfidenceVO:
        """
        Calcula o score de qualidade de uma face detectada.

        :param frame: Frame onde a face foi detectada.
        :param bbox: Bounding box da face.
        :param confidence: Confiança da detecção YOLO.
        :param landmarks: Landmarks faciais (pode ser None).
        :param peso_confianca: Peso para score de confiança (padrão: 3).
        :param peso_tamanho: Peso para score de tamanho (padrão: 4).
        :param peso_frontal: Peso para score de frontalidade (padrão: 6).
        :param peso_proporcao: Peso para score de proporção (padrão: 1).
        :param peso_nitidez: Peso para score de nitidez (padrão: 1).
        :return: Score de qualidade como ConfidenceVO (0.0 a 1.0).
        """
        # Usa pesos passados como parâmetros com valores padrão

        # Calcula scores individuais usando métodos internos
        score_confianca = FaceQualityService._calculate_confidence_score(confidence)
        score_tamanho = FaceQualityService._calculate_size_score(bbox, frame)
        score_frontal = FaceQualityService._calculate_frontal_score(landmarks)
        score_proporcao = FaceQualityService._calculate_proportion_score(bbox)
        score_nitidez = FaceQualityService._calculate_sharpness_score(frame, bbox)

        # Calcula score final ponderado
        total_peso = peso_confianca + peso_tamanho + peso_frontal + peso_proporcao + peso_nitidez

        score_final = (
            score_confianca * peso_confianca +
            score_tamanho * peso_tamanho +
            score_frontal * peso_frontal +
            score_proporcao * peso_proporcao +
            score_nitidez * peso_nitidez
        ) / total_peso

        return ConfidenceVO(score_final)
