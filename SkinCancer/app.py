# app.py - Enhanced PyTorch Streamlit application for skin disease classification

import streamlit as st
import torch
import pandas as pd
from PIL import Image
from torchvision import models
from helpers import preprocess_image, generate_gradcam, apply_heatmap, get_prediction, generate_pdf_report

CLASSES = [
    'Actinic Keratoses (Solar Keratoses)',
    'Basal Cell Carcinoma',
    'Benign Keratosis-like Lesions',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic Nevi',
    'Vascular Lesions'
]

DESCRIPTIONS = {
    'Actinic Keratoses (Solar Keratoses)': "Rough, scaly patches from sun exposure. Early form of skin cancer that can develop into squamous cell carcinoma if left untreated.",
    'Basal Cell Carcinoma': "Most common skin cancer. Appears as pearl-like bump or patch. Rarely spreads but can cause local damage if not treated.",
    'Benign Keratosis-like Lesions': "Harmless growths that may resemble cancer but are not malignant. Include seborrheic keratoses and solar lentigines.",
    'Dermatofibroma': "Small, hard benign skin growths, usually on legs. Often result from minor injuries like insect bites or splinters.",
    'Melanoma': "Most dangerous skin cancer. Early detection is critical as it can spread rapidly to other parts of the body.",
    'Melanocytic Nevi': "Common moles. Usually harmless but should be monitored for changes in size, shape, or color.",
    'Vascular Lesions': "Abnormal blood vessels in skin including hemangiomas and port-wine stains. Usually benign but may require treatment for cosmetic reasons."
}

# Medical reference links for each condition
MEDICAL_LINKS = {
    'Actinic Keratoses (Solar Keratoses)': [
        ("Mayo Clinic", "https://www.mayoclinic.org/diseases-conditions/actinic-keratosis/symptoms-causes/syc-20354969"),
        ("WebMD", "https://www.webmd.com/skin-problems-and-treatments/actinic-keratosis"),
        ("Healthline", "https://www.healthline.com/health/actinic-keratosis")
    ],
    'Basal Cell Carcinoma': [
        ("Mayo Clinic", "https://www.mayoclinic.org/diseases-conditions/basal-cell-carcinoma/symptoms-causes/syc-20354187"),
        ("WebMD", "https://www.webmd.com/melanoma-skin-cancer/basal-cell-carcinoma"),
        ("American Cancer Society", "https://www.cancer.org/cancer/basal-and-squamous-cell-skin-cancer.html")
    ],
    'Benign Keratosis-like Lesions': [
        ("Mayo Clinic", "https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878"),
        ("WebMD", "https://www.webmd.com/skin-problems-and-treatments/seborrheic-keratoses"),
        ("Healthline", "https://www.healthline.com/health/seborrheic-keratosis")
    ],
    'Dermatofibroma': [
        ("Mayo Clinic", "https://www.mayoclinic.org/diseases-conditions/dermatofibroma/symptoms-causes/syc-20354041"),
        ("WebMD", "https://www.webmd.com/skin-problems-and-treatments/what-is-dermatofibroma"),
        ("Healthline", "https://www.healthline.com/health/dermatofibroma")
    ],
    'Melanoma': [
        ("Mayo Clinic", "https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884"),
        ("WebMD", "https://www.webmd.com/melanoma-skin-cancer/melanoma-guide/melanoma-skin-cancer"),
        ("American Cancer Society", "https://www.cancer.org/cancer/melanoma-skin-cancer.html")
    ],
    'Melanocytic Nevi': [
        ("Mayo Clinic", "https://www.mayoclinic.org/diseases-conditions/moles/symptoms-causes/syc-20375200"),
        ("WebMD", "https://www.webmd.com/skin-problems-and-treatments/moles-freckles-skin-tags"),
        ("Healthline", "https://www.healthline.com/health/moles")
    ],
    'Vascular Lesions': [
        ("Mayo Clinic", "https://www.mayoclinic.org/diseases-conditions/hemangioma/symptoms-causes/syc-20352334"),
        ("WebMD", "https://www.webmd.com/skin-problems-and-treatments/birthmarks"),
        ("Healthline", "https://www.healthline.com/health/vascular-lesions")
    ]
}

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(model.last_channel, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, len(CLASSES)),
        torch.nn.Softmax(dim=1)
    )
    model.load_state_dict(torch.load("skin_disease_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def main():
    st.set_page_config(
        page_title="Skin Disease Classifier",
        page_icon="🔬",
        layout="wide"
    )
    st.title("🔬 Skin Disease Classification System")
    st.write("Upload a photo of a skin condition to get an AI diagnosis with detailed analysis")

    model = load_model()
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            image_tensor = preprocess_image(image)
            # Updated to get all probabilities
            predicted_class, confidence, class_index, all_probabilities = get_prediction(model, image_tensor, CLASSES)
            heatmap = generate_gradcam(model, image_tensor, class_index)
            superimposed = apply_heatmap(image, heatmap)

        with col2:
            st.subheader("Analysis Visualization")
            st.image(superimposed, caption="Areas of Interest (Grad-CAM)", use_column_width=True)

        # Diagnosis Results Section
        st.subheader("🎯 Diagnosis Results")
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.write(f"**Predicted Condition:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Confidence level indicator
            if confidence > 80:
                st.success("High Confidence")
            elif confidence > 60:
                st.warning("Moderate Confidence")
            else:
                st.error("Low Confidence - Consider additional evaluation")

        with col4:
            st.subheader("📊 All Class Probabilities")
            # Create bar chart of all probabilities
            prob_df = pd.DataFrame({
                'Condition': CLASSES,
                'Probability (%)': all_probabilities * 100
            })
            st.bar_chart(prob_df.set_index('Condition'))

        # Medical Information Section
        st.subheader("🩺 Medical Information")
        col5, col6 = st.columns([2, 1])
        
        with col5:
            st.write("**Description:**")
            st.write(DESCRIPTIONS.get(predicted_class, "No description available."))
            
        with col6:
            st.write("**Medical Resources:**")
            if predicted_class in MEDICAL_LINKS:
                for source_name, url in MEDICAL_LINKS[predicted_class]:
                    st.markdown(f"• [{source_name}]({url})")
            else:
                st.write("No links available for this condition.")

        # PDF Report Generation
        st.subheader("📄 Generate Report")
        col7, col8 = st.columns([1, 2])
        
        with col7:
            if st.button("Generate PDF Report"):
                try:
                    with st.spinner("Generating PDF report..."):
                        pdf_buffer = generate_pdf_report(
                            image, 
                            superimposed, 
                            predicted_class, 
                            confidence, 
                            DESCRIPTIONS.get(predicted_class, "No description available.")
                        )
                        
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"skin_disease_report_{predicted_class.replace(' ', '_').lower()}.pdf",
                            mime="application/pdf"
                        )
                        st.success("PDF report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.info("Please make sure reportlab is installed: pip install reportlab")

        with col8:
            st.info("The PDF report includes your uploaded image, diagnosis results, confidence score, AI visualization, and medical description.")

        # Disclaimer
        st.write("---")
        st.warning("**⚠️ Medical Disclaimer:** This tool is for educational purposes only and should not be used as a substitute for professional medical advice. Always consult with a qualified healthcare provider for proper diagnosis and treatment.")

    # Instructions and Information
    with st.expander("📋 How to use this tool"):
        st.write("""
        1. **Upload an image** of a skin condition (JPG, JPEG, or PNG format)
        2. **Wait for analysis** - The AI will process your image using deep learning
        3. **Review results** - Check the predicted diagnosis and confidence level
        4. **Examine probabilities** - View the bar chart showing likelihood of each condition
        5. **Read medical information** - Learn about the condition and access trusted sources
        6. **Generate report** - Download a PDF with all results for your records
        7. **Consult a doctor** - Always seek professional medical advice
        """)

    with st.expander("🔬 About this AI model"):
        st.write("""
        **Model Architecture:** MobileNetV2 (optimized for mobile and web deployment)
        
        **Training Data:** HAM10000 dataset - 10,000+ dermatoscopic images
        
        **Classes Detected:** 7 common skin conditions including melanoma, basal cell carcinoma, and benign lesions
        
        **Visualization:** Grad-CAM (Gradient-weighted Class Activation Mapping) shows which areas the AI focused on
        
        **Limitations:** This model is trained on dermatoscopic images and may not perform well on regular photos. Results should always be verified by medical professionals.
        """)

    with st.expander("⚕️ When to see a doctor"):
        st.write("""
        **Seek immediate medical attention if you notice:**
        - Asymmetrical moles or lesions
        - Irregular borders or color variations
        - Diameter larger than 6mm (size of a pencil eraser)
        - Evolving size, shape, or color
        - Bleeding, itching, or pain
        - Any new or changing skin growth
        
        **Remember:** Early detection saves lives, especially for melanoma.
        """)

if __name__ == "__main__":
    main()