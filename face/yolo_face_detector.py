import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
from emotrace_utils.config import FACE_DETECTION_CONFIG, FRAMES_CROPPED_DIR
from emotrace_utils.logger import get_logger

logger = get_logger(__name__)


class YOLOFaceDetector:
    """Face detector using YOLOv5 (CPU optimized)."""
    
    def __init__(self, model_name: str = FACE_DETECTION_CONFIG["model_name"]):
        logger.info(f"Initializing YOLOv5 face detector with model: {model_name}")
        
        self.device = torch.device('cpu')
        self.model = None
        self.conf_threshold = FACE_DETECTION_CONFIG["conf_threshold"]
        self.face_size = FACE_DETECTION_CONFIG["face_size"]
        
        try:
            logger.info(f"Loading YOLOv5 model: {model_name}")
            self.model = torch.hub.load('ultralytics/yolov5', model_name, force_reload=False)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"YOLOv5 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv5 model: {str(e)}")
            raise Exception(f"YOLOv5 model initialization failed: {str(e)}")
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in frame using YOLOv5.
        
        Returns:
            List of (x, y, w, h, confidence) tuples
        """
        if frame is None or frame.size == 0:
            return []
        
        if self.model is None:
            return []
        
        try:
            with torch.no_grad():
                results = self.model(frame)
            
            detections = []
            preds = results.pred[0]
            
            if preds.shape[0] == 0:
                return []
            
            for *box, conf, cls in preds:
                if conf >= self.conf_threshold:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    w = x2 - x1
                    h = y2 - y1
                    detections.append((x1, y1, w, h, float(conf)))
            
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
