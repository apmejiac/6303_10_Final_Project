# Imports
import streamlit as st
import torch
import io
from PIL import Image
from Toolbox import predict
# from Toolbox import predict_model # -- create model prediction file

# Creating different tabs for the presentation
def overview_tab():
    """
    Showing overview of the project and applications
    """
    
    st.title("Breast Cancer Image Classification Model")
    
    text = "write description here"
    st.write(text)
    
    # Upload example images, including regular, mask, and overlayed images
    class_distribtion = Image.open("class_distribution_plot.png")
    st.image(class_distribtion, caption="Distribution of Classes", use_column_width=False)
    
    st. divider()
    
    sample_images = Image.open("sample_images.png")
    st.image(sample_images, caption = "Sample Images from Dataset")
    
    overlayed_image = Image.open("Data/ImagesOverlay/malignant/malignant (1)_overlay.png")
    st.image(overlayed_image, caption="Sample Overlayed Image")
    
def metrics_tab():
    """
    Final metrics and results of model
    """
    st.title("Final Results and Metrics")
    
    # upload metrics of final model
    
    # Table of final metrics
    
    # Graph showing training and validation loss
    
    # Confusion matrix
    
def demo_tab():
    """
    Allow for a demo file to upload for the model to predict on
    """
    st.title("Demo: Predict on an Example image")
    
    # Upload an image for prediction
    image_upload = st.file_uploader("Select an image. . . ", type="png")
    
    if image_upload:
        image = Image.open(image_upload)
        st.image(image, caption="Uploaded Image to predict", use_column_width=False)
        
        # Placeholder prediction for now
        prediction = predict("trained_model.pt", image)
        
        st.write(f"Class prediction: {prediction}")
        
def main():
    st.set_page_config(page_title = "Breat Cancer Classification", layout='wide')
    
    # Creating tabs
    tabs = ["Project Overview", "Results", "Demo"]
    current_tab = st.sidebar.radio("Select Tab", tabs)
    
    if current_tab == "Project Overview":
        overview_tab()
    elif current_tab == "Results":
        metrics_tab()
    elif current_tab == "Demo":
        demo_tab()

if __name__ == "__main__":
    main()