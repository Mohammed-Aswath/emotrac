import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from emotrace_utils.logger import get_logger

logger = get_logger(__name__)


def plot_au_trajectory(df_aus: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """Plot AU intensities over time."""
    au_cols = [col for col in df_aus.columns if col.startswith('AU')]
    
    if not au_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No AU data available", ha="center", va="center")
        return fig, ax
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for au_col in au_cols[:10]:
        ax.plot(df_aus['frame_num'], df_aus[au_col].astype(float), label=au_col, alpha=0.7)
    
    ax.set_xlabel("Frame Number", fontsize=12)
    ax.set_ylabel("AU Intensity", fontsize=12)
    ax.set_title("Action Unit Trajectories Over Time", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax


def plot_emotion_distribution(df_aus: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """Plot emotion probabilities over time."""
    emotion_cols = [col for col in df_aus.columns if col.startswith('emotion_')]
    
    if not emotion_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No emotion data available", ha="center", va="center")
        return fig, ax
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for emotion_col in emotion_cols:
        ax.plot(df_aus['frame_num'], df_aus[emotion_col].astype(float), 
                label=emotion_col.replace('emotion_', '').capitalize(), alpha=0.8, linewidth=2)
    
    ax.set_xlabel("Frame Number", fontsize=12)
    ax.set_ylabel("Emotion Probability", fontsize=12)
    ax.set_title("Emotion Distribution Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    return fig, ax


def plot_micro_expressions(df_events: pd.DataFrame, total_frames: int) -> Tuple[plt.Figure, plt.Axes]:
    """Plot micro-expression timeline."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if df_events.empty:
        ax.text(0.5, 0.5, "No micro-expressions detected", ha="center", va="center", fontsize=12)
        ax.set_xlim([0, total_frames])
        ax.set_ylim([0, 1])
    else:
        y_pos = 0
        colors = plt.cm.tab20(np.linspace(0, 1, len(df_events)))
        
        for idx, (_, event) in enumerate(df_events.iterrows()):
            onset = int(event['onset_frame'])
            offset = int(event['offset_frame'])
            apex = int(event['apex_frame'])
            au = event['au']
            intensity = float(event['peak_intensity'])
            
            duration = offset - onset + 1
            ax.barh(y_pos, duration, left=onset, height=0.6, 
                   color=colors[idx], alpha=0.8, edgecolor='black', linewidth=1.5)
            
            ax.plot(apex, y_pos, marker='*', markersize=15, color='red', markeredgecolor='darkred', markeredgewidth=1)
            
            ax.text(onset - 2, y_pos, f"{au}\n({intensity:.0f})", 
                   ha='right', va='center', fontsize=9)
            
            y_pos += 1
        
        ax.set_xlim([0, total_frames])
        ax.set_ylim([-0.5, len(df_events) - 0.5])
    
    ax.set_xlabel("Frame Number", fontsize=12)
    ax.set_ylabel("Micro-Expression Event", fontsize=12)
    ax.set_title("Detected Micro-Expressions Timeline", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    return fig, ax
