from os import path
import streamlit as st
from PIL import Image
import torch

# Load the YOLOv5 model
model = torch.hub.load('yolov5-master', 'custom', path="model/best.pt", source="local")

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
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button('Detect Objects'):
            # Perform object detection
            results = detect_objects(image)

            # Display the results
            st.subheader('Detected Objects:')
            for obj in results.names[1:]:
                detections = results.pandas().xyxy[results.names.index(obj)]
                st.write(f"{obj}: {len(detections)}")
                for index, row in detections.iterrows():
                    st.write(f"Class: {obj}, Confidence: {row['confidence']:.2f}, "
                             f"Bounding Box: [{int(row['xmin'])}, {int(row['ymin'])}, "
                             f"{int(row['xmax'])}, {int(row['ymax'])}]")

if __name__ == '__main__':
    main()
