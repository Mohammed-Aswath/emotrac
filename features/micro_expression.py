import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from emotrace_utils.config import MICRO_EXPRESSION_CONFIG, MICRO_EVENTS_DIR
from emotrace_utils.logger import get_logger

logger = get_logger(__name__)


class MicroExpressionDetector:
    """Detect micro-expressions from AU time-series."""
    
    def __init__(
        self,
        au_change_threshold: float = MICRO_EXPRESSION_CONFIG["au_change_threshold"],
        min_duration: int = MICRO_EXPRESSION_CONFIG["min_duration_frames"],
        max_duration: int = MICRO_EXPRESSION_CONFIG["max_duration_frames"]
    ):
        self.au_change_threshold = au_change_threshold
        self.min_duration = min_duration
        self.max_duration = max_duration
    
    def _get_au_columns(self, df: pd.DataFrame) -> List[str]:
        """Get AU columns from dataframe."""
        return [col for col in df.columns if col.startswith('AU')]
    
    def _get_emotion_columns(self, df: pd.DataFrame) -> List[str]:
        """Get emotion columns from dataframe."""
        return [col for col in df.columns if col.startswith('emotion_')]
    
    def detect_rapid_changes(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect rapid AU changes (micro-expressions).
        
        Returns:
            List of event dictionaries with onset, apex, offset, AU, emotion
        """
        if df.empty or len(df) < 2:
            return []
        
        au_cols = self._get_au_columns(df)
        emotion_cols = self._get_emotion_columns(df)
        
        if not au_cols:
            return []
        
        events = []
        
        for au_col in au_cols:
            au_values = df[au_col].values.astype(float)
            
            delta_au = np.abs(np.diff(au_values))
            
            in_event = False
            event_start = None
            peak_idx = None
            peak_val = 0.0
            
            for i in range(len(delta_au)):
                if delta_au[i] > self.au_change_threshold:
                    if not in_event:
                        in_event = True
                        event_start = i
                        peak_idx = i
                        peak_val = au_values[i]
                    else:
                        if au_values[i] > peak_val:
                            peak_idx = i
                            peak_val = au_values[i]
                else:
                    if in_event:
                        event_duration = i - event_start
                        
                        if self.min_duration <= event_duration <= self.max_duration:
                            dominant_emotion = "neutral"
                            max_emotion_val = 0.0
                            
                            if emotion_cols and peak_idx < len(df):
                                for emotion_col in emotion_cols:
                                    emotion_val = float(df.iloc[peak_idx][emotion_col])
                                    if emotion_val > max_emotion_val:
                                        max_emotion_val = emotion_val
                                        dominant_emotion = emotion_col.replace('emotion_', '')
                            
                            event = {
                                "onset_frame": int(event_start),
                                "apex_frame": int(peak_idx),
                                "offset_frame": int(i),
                                "duration_frames": event_duration,
                                "au": au_col,
                                "peak_intensity": float(peak_val),
                                "dominant_emotion": dominant_emotion
                            }
                            events.append(event)
                        
                        in_event = False
                        event_start = None
                        peak_idx = None
                        peak_val = 0.0
        
        return events
    
    def save_events(self, events: List[Dict], clip_id: str) -> pd.DataFrame:
        """Save detected micro-expression events to CSV."""
        logger.info(f"Detected {len(events)} micro-expression events")
        
        if events:
            df_events = pd.DataFrame(events)
        else:
            df_events = pd.DataFrame(columns=[
                "onset_frame", "apex_frame", "offset_frame",
                "duration_frames", "au", "peak_intensity", "dominant_emotion"
            ])
        
        output_dir = MICRO_EVENTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{clip_id}_events.csv"
        
        df_events.to_csv(output_path, index=False)
        logger.info(f"Saved micro-expression events to {output_path}")
        
        return df_events
