import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from emotrace_utils.config import FACE_DETECTION_CONFIG, FRAMES_CROPPED_DIR
from emotrace_utils.logger import get_logger

logger = get_logger(__name__)


class YOLOFaceDetector:
    """Face detector using OpenCV Haar Cascade."""
    
    def __init__(self, model_name: str = FACE_DETECTION_CONFIG["model_name"]):
        logger.info(f"Initializing OpenCV face detector")
        
        self.conf_threshold = FACE_DETECTION_CONFIG["conf_threshold"]
        self.face_size = FACE_DETECTION_CONFIG["face_size"]
        
        try:
            # Try multiple methods to find Haar Cascade classifier
            cascade_filename = 'haarcascade_frontalface_default.xml'
            cascade_path = None
            
            # Method 1: Try OpenCV installation directory
            cv2_path = Path(cv2.__file__).parent
            candidate_path = cv2_path / 'data' / 'haarcascades' / cascade_filename
            if candidate_path.exists():
                cascade_path = str(candidate_path)
                logger.debug(f"Found cascade at: {cascade_path}")
            
            # Method 2: Try cv2.data.haarcascades if available
            if cascade_path is None:
                try:
                    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                        cascade_path = cv2.data.haarcascades + cascade_filename
                        if not Path(cascade_path).exists():
                            cascade_path = None
                except:
                    pass
            
            # Method 3: Try common system paths
            if cascade_path is None:
                possible_paths = [
                    cv2_path / 'share' / 'opencv4' / 'haarcascades' / cascade_filename,
                    cv2_path / 'share' / 'opencv' / 'haarcascades' / cascade_filename,
                ]
                for path in possible_paths:
                    if path.exists():
                        cascade_path = str(path)
                        logger.debug(f"Found cascade at: {cascade_path}")
                        break
            
            if cascade_path is None:
                raise Exception(f"Could not locate {cascade_filename} in OpenCV installation. Please run 'python download_cascade.py' to download cascade files.")
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise Exception(f"Failed to load Haar Cascade classifier from {cascade_path}")
            
            logger.info(f"OpenCV Haar Cascade face detector loaded from {cascade_path}")
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {str(e)}")
            raise Exception(f"Face detector initialization failed: {str(e)}")
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in frame using OpenCV Haar Cascade.
        
        Returns:
            List of (x, y, w, h, confidence) tuples
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                maxSize=(500, 500)
            )
            
            detections = []
            for (x, y, w, h) in faces:
                confidence = 0.9  # Haar Cascade doesn't provide confidence, use default
                detections.append((x, y, w, h, confidence))
            
            return detections
        
        except Exception as e:
            logger.error(f"Error during face detection: {str(e)}")
            return []
    
    def get_best_face(self, detections: List[Tuple[int, int, int, int, float]]) -> Optional[Tuple[int, int, int, int]]:
        """Select highest-confidence face."""
        if not detections:
            return None
        
        best = max(detections, key=lambda x: x[4])
        return (best[0], best[1], best[2], best[3])
    
    def crop_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Crop and resize face to standard size."""
        if bbox is None:
            return None
        
        x, y, w, h = bbox
        x = max(0, x)
        y = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        
        if x2 <= x or y2 <= y:
            return None
        
        face_crop = frame[y:y2, x:x2]
        face_resized = cv2.resize(face_crop, (self.face_size, self.face_size))
        
        return face_resized
    
    def process_frame(self, frame: np.ndarray, clip_id: str, frame_num: int) -> Optional[str]:
        """
        Detect, crop, and save face.
        
        Returns:
            Path to cropped face image or None
        """
        detections = self.detect_faces(frame)
        
        if not detections:
            logger.debug(f"No faces detected in frame {frame_num}")
            return None
        
        best_bbox = self.get_best_face(detections)
        cropped = self.crop_face(frame, best_bbox)
        
        if cropped is None:
            return None
        
        output_dir = FRAMES_CROPPED_DIR / clip_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"face_{frame_num:04d}.jpg"
        cv2.imwrite(str(output_path), cropped)
        
        logger.debug(f"Saved cropped face to {output_path}")
        return str(output_path)
