#!/usr/bin/env python
"""
Validate that all project requirements are met.
Checks for completeness, consistency, and functionality.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from emotrace_utils.logger import get_logger

logger = get_logger(__name__)


class ProjectValidator:
    """Validate project structure and completeness."""
    
    REQUIRED_FILES = {
        "root": [
            "app.py",
            "run_pipeline.py",
            "requirements.txt",
            "README.md",
            "QUICKSTART.md",
            "quickstart.py",
            "validate_project.py"
        ],
        "utils": [
            "config.py",
            "logger.py",
            "__init__.py"
        ],
        "video": [
            "extract_frames.py",
            "__init__.py"
        ],
        "face": [
            "yolo_face_detector.py",
            "__init__.py"
        ],
        "features": [
            "au_extractor.py",
            "micro_expression.py",
            "__init__.py"
        ],
        "scoring": [
            "depression_screener.py",
            "feature_engineering.py",
            "recommendation.py",
            "__init__.py"
        ],
        "visualization": [
            "plots.py",
            "__init__.py"
        ],
        "data": [
            "raw_videos/",
            "frames/",
            "frames_cropped/",
            "au_results/",
            "micro_events/"
        ]
    }
    
    REQUIRED_CLASSES = {
        "video.extract_frames": ["extract_frames"],
        "face.yolo_face_detector": ["YOLOFaceDetector"],
        "features.au_extractor": ["AUExtractor"],
        "features.micro_expression": ["MicroExpressionDetector"],
        "scoring.feature_engineering": ["FeatureEngineer"],
        "scoring.depression_screener": ["DepressionScreener"],
        "scoring.recommendation": ["RecommendationEngine"],
        "visualization.plots": [
            "plot_au_trajectory",
            "plot_emotion_distribution",
            "plot_micro_expressions"
        ]
    }
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.passed = 0
        self.failed = 0
    
    def check_files(self):
        """Check all required files exist."""
        logger.info("Checking file structure...")
        
        for directory, files in self.REQUIRED_FILES.items():
            if directory == "root":
                dir_path = self.root_dir
            else:
                dir_path = self.root_dir / directory
            
            for file in files:
                file_path = dir_path / file
                if file_path.exists():
                    logger.info(f"  ✓ {directory}/{file}")
                    self.passed += 1
                else:
                    logger.error(f"  ✗ {directory}/{file} NOT FOUND")
                    self.failed += 1
    
    def check_implementations(self):
        """Check that classes and functions are implemented (not placeholders)."""
        logger.info("\nChecking implementations...")
        
        checks = [
            ("video.extract_frames", "extract_frames", "def extract_frames"),
            ("face.yolo_face_detector", "YOLOFaceDetector", "class YOLOFaceDetector"),
            ("features.au_extractor", "AUExtractor", "class AUExtractor"),
            ("features.micro_expression", "MicroExpressionDetector", "class MicroExpressionDetector"),
            ("scoring.feature_engineering", "FeatureEngineer", "class FeatureEngineer"),
            ("scoring.depression_screener", "DepressionScreener", "class DepressionScreener"),
            ("scoring.recommendation", "RecommendationEngine", "class RecommendationEngine"),
            ("visualization.plots", "plot_au_trajectory", "def plot_au_trajectory"),
        ]
        
        for module_name, class_name, search_str in checks:
            module_path = self.root_dir / module_name.replace(".", "/") + ".py"
            
            if module_path.exists():
                with open(module_path, 'r') as f:
                    content = f.read()
                
                if search_str in content:
                    if "pass" not in content.split(search_str)[1].split('\n')[0:5]:
                        logger.info(f"  ✓ {class_name} implemented")
                        self.passed += 1
                    else:
                        logger.error(f"  ✗ {class_name} has placeholder (pass)")
                        self.failed += 1
                else:
                    logger.error(f"  ✗ {class_name} not found")
                    self.failed += 1
            else:
                logger.error(f"  ✗ Module {module_name} not found")
                self.failed += 1
    
    def check_streamlit_app(self):
        """Check that Streamlit app is implemented."""
        logger.info("\nChecking Streamlit app...")
        
        app_path = self.root_dir / "app.py"
        
        if app_path.exists():
            with open(app_path, 'r') as f:
                content = f.read()
            
            checks = [
                ("import streamlit", "Streamlit import"),
                ("st.file_uploader", "File upload widget"),
                ("st.button", "Run button"),
                ("st.metric", "Metrics display"),
                ("st.pyplot", "Plot display"),
                ("st.tabs", "Tabbed interface"),
            ]
            
            for search_str, description in checks:
                if search_str in content:
                    logger.info(f"  ✓ {description}")
                    self.passed += 1
                else:
                    logger.error(f"  ✗ {description} missing")
                    self.failed += 1
        else:
            logger.error("  ✗ app.py not found")
            self.failed += 1
    
    def check_pipeline(self):
        """Check that main pipeline is implemented."""
        logger.info("\nChecking main pipeline...")
        
        pipeline_path = self.root_dir / "run_pipeline.py"
        
        if pipeline_path.exists():
            with open(pipeline_path, 'r') as f:
                content = f.read()
            
            checks = [
                ("def run_analysis_pipeline", "Pipeline function"),
                ("extract_frames", "Frame extraction"),
                ("YOLOFaceDetector", "Face detection"),
                ("AUExtractor", "AU extraction"),
                ("MicroExpressionDetector", "Micro-expression detection"),
                ("FeatureEngineer", "Feature engineering"),
                ("DepressionScreener", "Risk scoring"),
                ("RecommendationEngine", "Recommendation generation"),
            ]
            
            for search_str, description in checks:
                if search_str in content:
                    logger.info(f"  ✓ {description}")
                    self.passed += 1
                else:
                    logger.error(f"  ✗ {description} missing")
                    self.failed += 1
        else:
            logger.error("  ✗ run_pipeline.py not found")
            self.failed += 1
    
    def check_no_placeholder_code(self):
        """Check that project doesn't have placeholder patterns."""
        logger.info("\nChecking for placeholder code...")
        
        placeholder_patterns = [
            ("pass", "Empty pass statement"),
            ("TODO", "TODO comment"),
            ("FIXME", "FIXME comment"),
            ("placeholder", "Placeholder text (case-insensitive)"),
        ]
        
        py_files = list(self.root_dir.rglob("*.py"))
        
        for py_file in py_files:
            if "__pycache__" in str(py_file):
                continue
            
            with open(py_file, 'r') as f:
                content = f.read()
            
            has_issues = False
            
            if " pass\n" in content or content.endswith("pass"):
                for line in content.split('\n'):
                    if line.strip() == "pass" and "except" not in content[max(0, content.find(line)-100):content.find(line)]:
                        logger.warning(f"  ! {py_file.name}: contains empty pass")
                        has_issues = True
                        break
            
            for pattern, description in placeholder_patterns[1:]:
                if pattern in content.upper():
                    logger.warning(f"  ! {py_file.name}: contains {description}")
                    has_issues = True
            
            if not has_issues:
                logger.info(f"  ✓ {py_file.relative_to(self.root_dir)}: No placeholders")
                self.passed += 1
    
    def run_validation(self):
        """Run all validations."""
        logger.info("="*60)
        logger.info("EmoTrace Project Validation")
        logger.info("="*60)
        
        self.check_files()
        self.check_implementations()
        self.check_streamlit_app()
        self.check_pipeline()
        self.check_no_placeholder_code()
        
        logger.info("\n" + "="*60)
        logger.info(f"Results: {self.passed} passed, {self.failed} failed")
        logger.info("="*60)
        
        if self.failed == 0:
            logger.info("\n✓ Project validation PASSED - Ready to use!")
            return True
        else:
            logger.error(f"\n✗ Project validation FAILED - {self.failed} issues found")
            return False


if __name__ == "__main__":
    validator = ProjectValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)
