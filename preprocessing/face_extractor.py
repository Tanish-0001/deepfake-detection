"""
Face extraction module using RetinaFace for face detection, alignment, and cropping.
"""

import os
# Fix TensorFlow/Keras 3 compatibility issue with retinaface
# Must be set before importing tensorflow
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class FaceDetection:
    """Container for face detection results."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5 facial landmarks
    aligned_face: Optional[np.ndarray] = None
    cropped_face: Optional[np.ndarray] = None


class FaceExtractor(ABC):
    """Abstract base class for face extractors."""
    
    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in an image.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of FaceDetection objects
        """
        pass
    
    @abstractmethod
    def extract_face(
        self, 
        image: np.ndarray,
        output_size: Tuple[int, int] = (224, 224),
        enlargement_factor: float = 1.3
    ) -> Optional[np.ndarray]:
        """
        Extract the primary face from an image.
        
        Args:
            image: BGR image as numpy array
            output_size: Target output size (width, height)
            enlargement_factor: Factor to enlarge bounding box
            
        Returns:
            Cropped and resized face image, or None if no face detected
        """
        pass


class RetinaFaceExtractor(FaceExtractor):
    """Face extractor using RetinaFace model."""
    
    def __init__(
        self, 
        detection_threshold: float = 0.9,
        device: str = "cuda"
    ):
        """
        Initialize RetinaFace extractor.
        
        Args:
            detection_threshold: Minimum confidence for face detection
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.detection_threshold = detection_threshold
        self.device = device
        self._model = None
        
    def _load_model(self):
        """Lazy load the RetinaFace model."""
        if self._model is None:
            try:
                from retinaface import RetinaFace
                self._model = RetinaFace
            except ImportError:
                raise ImportError(
                    "RetinaFace is not installed. "
                    "Install it with: pip install retinaface"
                )
        return self._model
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect all faces in an image using RetinaFace.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of FaceDetection objects sorted by confidence
        """
        model = self._load_model()
        
        # Convert BGR to RGB for RetinaFace
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = model.detect_faces(image_rgb, threshold=self.detection_threshold)
        
        if not faces:
            return []
        
        detections = []
        for face_key, face_data in faces.items():
            bbox = np.array(face_data['facial_area'])  # [x1, y1, x2, y2]
            confidence = face_data['score']
            
            # Extract landmarks if available
            landmarks = None
            if 'landmarks' in face_data:
                landmarks = np.array([
                    face_data['landmarks']['left_eye'],
                    face_data['landmarks']['right_eye'],
                    face_data['landmarks']['nose'],
                    face_data['landmarks']['mouth_left'],
                    face_data['landmarks']['mouth_right']
                ])
            
            detections.append(FaceDetection(
                bbox=bbox,
                confidence=confidence,
                landmarks=landmarks
            ))
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        return detections
    
    def align_face(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        output_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Align face based on eye landmarks.
        
        Args:
            image: BGR image
            landmarks: 5-point facial landmarks
            output_size: Target output size
            
        Returns:
            Aligned face image
        """
        # Get eye centers
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # Calculate angle between eyes
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Calculate center between eyes
        eye_center = (
            (left_eye[0] + right_eye[0]) // 2,
            (left_eye[1] + right_eye[1]) // 2
        )
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        
        # Apply rotation
        aligned = cv2.warpAffine(
            image, M, (image.shape[1], image.shape[0]),
            flags=cv2.INTER_CUBIC
        )
        
        return aligned
    
    def enlarge_bbox(
        self,
        bbox: np.ndarray,
        image_shape: Tuple[int, int],
        enlargement_factor: float = 1.3
    ) -> np.ndarray:
        """
        Enlarge bounding box by a factor while keeping it within image bounds.
        
        Args:
            bbox: Original bounding box [x1, y1, x2, y2]
            image_shape: Image shape (height, width)
            enlargement_factor: Factor to enlarge the bbox
            
        Returns:
            Enlarged bounding box
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate center and size
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Enlarge
        new_width = width * enlargement_factor
        new_height = height * enlargement_factor
        
        # Calculate new coordinates
        new_x1 = int(max(0, center_x - new_width / 2))
        new_y1 = int(max(0, center_y - new_height / 2))
        new_x2 = int(min(image_shape[1], center_x + new_width / 2))
        new_y2 = int(min(image_shape[0], center_y + new_height / 2))
        
        return np.array([new_x1, new_y1, new_x2, new_y2])
    
    def crop_and_resize(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        output_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Crop face region and resize to target size.
        
        Args:
            image: BGR image
            bbox: Bounding box [x1, y1, x2, y2]
            output_size: Target output size (width, height)
            
        Returns:
            Cropped and resized face image
        """
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Crop
        cropped = image[y1:y2, x1:x2]
        
        # Handle edge case of empty crop
        if cropped.size == 0:
            return np.zeros((*output_size[::-1], 3), dtype=np.uint8)
        
        # Resize
        resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def extract_face(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int] = (224, 224),
        enlargement_factor: float = 1.3,
        align: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract the primary (largest confidence) face from an image.
        
        Args:
            image: BGR image as numpy array
            output_size: Target output size (width, height)
            enlargement_factor: Factor to enlarge bounding box
            align: Whether to align face based on eye landmarks
            
        Returns:
            Cropped and resized face image, or None if no face detected
        """
        detections = self.detect_faces(image)
        
        if not detections:
            return None
        
        # Get the face with highest confidence
        best_detection = detections[0]
        
        # Optionally align face
        if align and best_detection.landmarks is not None:
            image = self.align_face(image, best_detection.landmarks, output_size)
            # Re-detect on aligned image for better bbox
            new_detections = self.detect_faces(image)
            if new_detections:
                best_detection = new_detections[0]
        
        # Enlarge bounding box
        enlarged_bbox = self.enlarge_bbox(
            best_detection.bbox,
            image.shape[:2],
            enlargement_factor
        )
        
        # Crop and resize
        face = self.crop_and_resize(image, enlarged_bbox, output_size)
        
        return face
    
    def extract_all_faces(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int] = (224, 224),
        enlargement_factor: float = 1.3,
        align: bool = True,
        max_faces: int = 1
    ) -> List[np.ndarray]:
        """
        Extract all detected faces from an image.
        
        Args:
            image: BGR image as numpy array
            output_size: Target output size (width, height)
            enlargement_factor: Factor to enlarge bounding box
            align: Whether to align faces
            max_faces: Maximum number of faces to extract
            
        Returns:
            List of cropped and resized face images
        """
        detections = self.detect_faces(image)
        
        if not detections:
            return []
        
        faces = []
        for detection in detections[:max_faces]:
            # Optionally align face
            working_image = image.copy()
            if align and detection.landmarks is not None:
                working_image = self.align_face(working_image, detection.landmarks, output_size)
                # Re-detect on aligned image
                new_detections = self.detect_faces(working_image)
                if new_detections:
                    detection = new_detections[0]
            
            # Enlarge bounding box
            enlarged_bbox = self.enlarge_bbox(
                detection.bbox,
                working_image.shape[:2],
                enlargement_factor
            )
            
            # Crop and resize
            face = self.crop_and_resize(working_image, enlarged_bbox, output_size)
            faces.append(face)
        
        return faces


def create_face_extractor(
    detector_type: str = "retinaface",
    **kwargs
) -> FaceExtractor:
    """
    Factory function to create face extractor.
    
    Args:
        detector_type: Type of detector ('retinaface', 'mtcnn', etc.)
        **kwargs: Additional arguments for the detector
        
    Returns:
        FaceExtractor instance
    """
    if detector_type.lower() == "retinaface":
        return RetinaFaceExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
