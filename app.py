# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper






# Function to set page configuration

def set_page_layout():
    st.set_page_config(
        page_title="Object Detection using YOLOv8",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    
# Function to display main title
def display_main_title():
    st.title("Object Detection And Tracking using YOLOv8")

# Function to display ML model configuration sidebar
def display_model_config_sidebar():
    st.sidebar.header("ML Model Config")
    model_type = st.sidebar.radio(
        "Select Task", ['Detection', 'Segmentation'])
    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 25, 100, 40)) / 100
    return model_type, confidence

# Function to load pre-trained ML model
def load_pretrained_model(model_type):
    if model_type == 'Detection':
        model_path = Path(settings.DETECTION_MODEL)
    elif model_type == 'Segmentation':
        model_path = Path(settings.SEGMENTATION_MODEL)
    try:
        model = helper.load_model(model_path)
        return model
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
        return None

# Function to display image/video configuration sidebar
def display_image_video_config_sidebar():
    st.sidebar.header("Image/Video Config")
    source_radio = st.sidebar.radio(
        "Select Source", settings.SOURCES_LIST)
    return source_radio

# Function to display uploaded image and detect objects
def display_uploaded_image_and_detection(model, confidence, source_img):
    col1, col2 = st.columns(2)
    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)
    with col2:
        if source_img is not None:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")


# Main function
def main():
    set_page_layout()
    display_main_title()
    model_type, confidence = display_model_config_sidebar()
    model = load_pretrained_model(model_type)
    if model is None:
        return
    source_radio = display_image_video_config_sidebar()
    if source_radio == settings.IMAGE:
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
        display_uploaded_image_and_detection(model, confidence, source_img)
    elif source_radio == settings.VIDEO:
        helper.play_stored_video(confidence, model)
    elif source_radio == settings.WEBCAM:
        helper.play_webcam(confidence, model)
    elif source_radio == settings.RTSP:
        helper.play_rtsp_stream(confidence, model)
    elif source_radio == settings.YOUTUBE:
        helper.play_youtube_video(confidence, model)
    else:
        st.error("Please select a valid source type!")
    

if __name__ == "__main__":
    main()

