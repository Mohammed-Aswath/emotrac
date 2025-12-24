import streamlit as st
import tempfile
from pathlib import Path
from run_pipeline import run_analysis_pipeline
from emotrace_utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="EmoTrace - Depression Risk Screener", layout="wide")

st.title("🎬 EmoTrace")
st.write("**Facial Expression Analysis for Depression Risk Screening**")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📹 Upload Video")
    uploaded_file = st.file_uploader("Drag and drop a video file (.mp4)", type=["mp4"])
    
    if uploaded_file is not None:
        st.success(f"✓ File selected: {uploaded_file.name}")

with col2:
    st.header("⚙️ Analysis Parameters")
    st.info("Using default analysis parameters")

st.divider()

if uploaded_file is not None:
    col_btn1, col_btn2 = st.columns([1, 1])
    
    with col_btn1:
        run_button = st.button("🔴 Run Analysis", use_container_width=True, type="primary")
    
    with col_btn2:
        st.button("Clear Results", use_container_width=True)
    
    if run_button:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_video_path = tmp_file.name
        
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        def progress_callback(step: int, message: str, percent: int):
            steps = [
                "Extracting frames",
                "Detecting faces",
                "Extracting action units",
                "Detecting micro-expressions",
                "Computing features",
                "Calculating risk score",
                "Generating recommendations",
                "Creating visualizations",
                "Saving results",
                "Completed"
            ]
            
            step_name = steps[step - 1] if 0 < step <= len(steps) else "Processing"
            
            with progress_placeholder.container():
                st.write(f"**Step {step}/10:** {step_name}")
                st.progress(percent / 100.0)
                st.caption(message)
        
        try:
            logger.info(f"Starting analysis on uploaded file: {uploaded_file.name}")
            
            result = run_analysis_pipeline(tmp_video_path, progress_callback=progress_callback)
            
            if result['status'] == 'success':
                st.success("✅ Analysis completed successfully!", icon="✅")
                
                st.divider()
                st.header("📊 Analysis Results")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Risk Score", f"{result['risk_score']:.1f}/100")
                
                with metric_col2:
                    risk_color = {
                        'LOW': '🟢',
                        'MEDIUM': '🟡',
                        'HIGH': '🔴'
                    }
                    st.metric("Risk Category", f"{risk_color.get(result['risk_band'], '')} {result['risk_band']}")
                
                with metric_col3:
                    st.metric("Frames Analyzed", result['num_frames'])
                
                st.divider()
                
                col_faces, col_detected = st.columns(2)
                with col_faces:
                    st.metric("Faces Detected", result['faces_detected'])
                with col_detected:
                    st.metric("Detection Rate", f"{(result['faces_detected']/result['num_frames']*100):.1f}%")
                
                st.divider()
                st.header("💡 Recommendation")
                
                st.info(result['recommendation'])
                
                st.divider()
                st.header("📈 Visualizations")
                
                if 'plots' in result and result['plots']:
                    plots = result['plots']
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    if 'au_plot' in plots:
                        with viz_col1:
                            st.pyplot(plots['au_plot'], use_container_width=True)
                    
                    if 'emotion_plot' in plots:
                        with viz_col2:
                            st.pyplot(plots['emotion_plot'], use_container_width=True)
                    
                    if 'micro_plot' in plots:
                        st.pyplot(plots['micro_plot'], use_container_width=True)
                
                with st.expander("⚠️ DISCLAIMER", expanded=True):
                    st.warning(
                        """
                        **IMPORTANT DISCLAIMER:** This is a research prototype for non-diagnostic facial expression analysis. 
                        This tool is NOT designed for medical diagnosis and cannot replace professional mental health assessment. 
                        Results should not be used for clinical decision-making. For mental health concerns, 
                        always consult with qualified healthcare professionals.
                        """
                    )
            
            else:
                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            logger.error(f"Analysis error: {str(e)}")
        
        finally:
            Path(tmp_video_path).unlink(missing_ok=True)

else:
    st.info("👆 Upload a video file to begin analysis")
