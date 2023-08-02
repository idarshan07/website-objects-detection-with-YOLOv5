from os import path
import streamlit as st
from PIL import Image
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path="model/best.pt")
# Function to perform object detection
def detect_objects(image):
    results = model(image)
    return results

# Streamlit app
def main():
    st.title("Website Object Detection using YOLOv5")
    st.write('Upload a Screenshot of any website and the app will detect objects.')


    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform inference and get results
        results = model(image)

        # Overlay bounding boxes on the image
        annotated_image = results.render()[0]
        
        st.subheader('Image with Detected Objects:')

        st.image(annotated_image, caption="Annotated Image", use_column_width=True)

        

if __name__ == '__main__':
    main()
