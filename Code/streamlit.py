# Imports
import streamlit as st
import torch
import io
from PIL import Image
from Toolbox import predict
import pandas as pd
# from Toolbox import predict_model # -- create model prediction file

# Creating different tabs for the presentation
def overview_tab():
    """
    Showing overview of the project and applications
    """
    
    st.title("Breast Cancer Image Classification Model")
    
    
    st.image("Breast_fibroadenoma.gif")
    
    text = "Ultrasound images can be used as a screening technique to diagnose breast cancer. They are highly accurate, with around 80% of breast cancers being detected"
    st.write(text)
    
    text = "Data is collected from Kaggle and contains 780 images labelled as Normal, Benign, or Malignant"
    st.write(text)
    
    text = "Our goal was to use machine learning to correctly classify these images into their respective categories"
    st.write(text)
    
    st.divider()
    # Upload example images, including regular, mask, and overlayed images
    st.write("Distribution of classes in original dataset")
    class_distribtion = Image.open("class_distribution_plot.png")
    st.image(class_distribtion, caption="Distribution of Classes", use_column_width=False)
    
    st.divider()
    
    st.write("Example images from dataset")
    sample_images = Image.open("sample_images_screenshot.png")
    st.image(sample_images, caption = "Sample Images from Dataset")
    
    st.divider()
    
    st.write("Exam of Overlayed image used for processing in alternative model")
    overlayed_image = Image.open("Data/ImagesOverlay/malignant/malignant (1)_overlay.png")
    st.image(overlayed_image, caption="Sample Overlayed Image")
    
def metrics_tab():
    """
    Final metrics and results of model
    """
    st.title("Final Results and Metrics")
    
    selected_tab = st.selectbox("Select a Model", ["Original Model", "Mask Overlay Model"], key="selectbox")
    
    st.markdown(
        """
        <style>
            .stSelectbox {
                margin-top: 20px;
            }
            .st-5c8- {
                width: 50px; /* Set the desired width */
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    if selected_tab == "Original Model":
        st.write("Metrics (Macro)")
        
        metrics = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [0.7265, 0.7372, 0.7265, 0.7295]
        }
        metrics_df = pd.DataFrame(metrics)
        st.table(metrics_df)

        st.write("Metrics by Class")
        metrics_class = {
            "Precision": [0.83, 0.60, 0.67],
            "Recall": [0.74, 0.66, 0.77],
            "F1 Score": [0.78, 0.62, 0.72]
        }
        class_df = pd.DataFrame(metrics_class, index=["Benign", "Malignant", "Normal"])
        st.table(class_df)
        
        # Confusion Matrix
        st.write("Confusion Matrix")
        confusion_matrix = "cm_original.png"
        cm_image = Image.open(confusion_matrix)
        st.image(cm_image, caption="Confusion Matrix", width=600)
        
        # Training loss
        st.write("Test Loss v. Validation Loss")
        loss_graph = "loss_original.png"
        loss_image = Image.open(loss_graph)
        st.image(loss_image, caption="Test v. Validation Loss", width=600)
        
        
    elif selected_tab == "Mask Overlay Model":
        st.write("Metrics (Macro)")
        
        metrics = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [0.9786, 0.9786, 0.9786, 0.9786]
        }
        metrics_df = pd.DataFrame(metrics)
        st.table(metrics_df)

        st.write("Metrics by Class")
        metrics_class = {
            "Precision": [0.98, 0.97, 1.00],
            "Recall": [0.99, 0.95, 0.96],
            'F1 Score': [0.98, 0.96, 1.00]
        }
        class_df = pd.DataFrame(metrics_class, index=["Benign", "Malignant", "Normal"])
        st.table(class_df)
        
        # Confusion Matrix
        st.write("Confusion Matrix")
        confusion_matrix = "cm_overlay.png"
        cm_image = Image.open(confusion_matrix)
        st.image(cm_image, caption="Confusion Matrix", width=600)
        
        # Training loss
        st.write("Test Loss v. Validation Loss")
        loss_graph = "loss_original.png"
        loss_image = Image.open(loss_graph)
        st.image(loss_image, caption="Test v. Validation Loss", width=600)
    
def demo_tab():
    """
    Allow for a demo file to upload for the model to predict on
    """
    st.title("Demo: Predict on an Example image")
    
    selected_tab = st.selectbox("Select a Model", ["Original Model", "Mask Overlay Model"], key="selectbox")
    
    if selected_tab == "Original Model":
        # Upload an image for prediction
        image_upload = st.file_uploader("Select an image. . . ", type="png")
        
        if image_upload:
            image = Image.open(image_upload)
            st.image(image, caption="Uploaded Image to predict", use_column_width=False)
            
            # Placeholder prediction for now
            prediction = predict("trained_model_original.pt", image)
            
            st.write(f"Class prediction: {prediction}")
    
    if selected_tab == "Mask Overlay Model":
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
    
    st.markdown(
    """
    <style>
        body {
            background-color: #f0f0f0; /* Light pink background color */
        }
        h1 {
            color: #ff3399; /* Pink text color for headers */
        }
        p {
            color: #ff0066; /* Pink text color for paragraphs */
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    
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