# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
import av
from typing import Any
import cv2
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class VideoProcessor(VideoTransformerBase):
    """Custom video processor for real-time webcam inference."""
    def __init__(self, model, conf, iou, selected_ind, enable_trk):
        self.model = model
        self.conf = conf
        self.iou = iou
        self.selected_ind = selected_ind
        self.enable_trk = enable_trk

    def recv(self, frame):
        """Process each video frame."""
        img = frame.to_ndarray(format="bgr24")
        
        if self.enable_trk == "Yes":
            results = self.model.track(
                img, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
            )
        else:
            results = self.model(img, conf=self.conf, iou=self.iou, classes=self.selected_ind)

        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

class Inference:
    """Main inference class with Streamlit integration."""
    def __init__(self, **kwargs: Any):
        check_requirements("streamlit>=1.29.0")
        import streamlit as st
        
        self.st = st
        self.source = None
        self.enable_trk = False
        self.conf = 0.25
        self.iou = 0.45
        self.org_frame = None
        self.ann_frame = None
        self.vid_file_name = None
        self.selected_ind = []
        self.model = None
        self.temp_dict = {"model": None, **kwargs}
        self.model_path = self.temp_dict["model"] if self.temp_dict["model"] else None

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        """Configure Streamlit UI elements."""
        menu_style = """<style>MainMenu {visibility: hidden;}</style>"""
        main_title = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""
        sub_title = """<div><h4 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-top:-15px; margin-bottom:50px;">Real-time object detection with Ultralytics YOLO! 🚀</h4></div>"""

        self.st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        self.st.markdown(menu_style, unsafe_allow_html=True)
        self.st.markdown(main_title, unsafe_allow_html=True)
        self.st.markdown(sub_title, unsafe_allow_html=True)

    def sidebar(self):
        """Configure sidebar elements."""
        with self.st.sidebar:
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)
            self.st.title("User Configuration")
            self.source = self.st.selectbox("Video Source", ("webcam", "video"))
            self.enable_trk = self.st.radio("Enable Tracking", ("Yes", "No"))
            self.conf = float(self.st.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01))
            self.iou = float(self.st.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))

        col1, col2 = self.st.columns(2)
        self.org_frame = col1.empty()
        self.ann_frame = col2.empty()

    def source_upload(self):
        """Handle video file uploads."""
        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"])
            if vid_file:
                g = io.BytesIO(vid_file.read())
                with open("temp_video.mp4", "wb") as out:
                    out.write(g.read())
                self.vid_file_name = "temp_video.mp4"

    def configure(self):
        """Configure model and classes."""
        available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path:
            available_models.insert(0, self.model_path.split(".pt")[0])
            
        selected_model = self.st.sidebar.selectbox("Model", available_models)
        with self.st.spinner("Loading model..."):
            self.model = YOLO(f"{selected_model.lower()}.pt")
            class_names = list(self.model.names.values())
        self.st.success("Model loaded!")
        
        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]
        self.selected_ind = list(self.selected_ind) if not isinstance(self.selected_ind, list) else self.selected_ind

    def inference(self):
        """Main inference pipeline."""
        self.web_ui()
        self.sidebar()
        
        if self.source == "webcam":
            self.configure()
            ctx = webrtc_streamer(
                key="yolo-inference",
                video_processor_factory=lambda: VideoProcessor(
                    self.model,
                    self.conf,
                    self.iou,
                    self.selected_ind,
                    self.enable_trk
                ),
                media_stream_constraints={"video": True, "audio": False},
            )
            
        elif self.source == "video":
            self.source_upload()
            self.configure()
            if self.st.sidebar.button("Start"):
                stop_button = self.st.button("Stop")
                cap = cv2.VideoCapture(self.vid_file_name)
                
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        self.st.warning("End of video stream")
                        break
                    
                    results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                    annotated_frame = results[0].plot()
                    
                    if stop_button:
                        cap.release()
                        self.st.stop()
                        
                    self.org_frame.image(frame, channels="BGR")
                    self.ann_frame.image(annotated_frame, channels="BGR")
                
                cap.release()
                cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else None
    Inference(model=model).inference()
