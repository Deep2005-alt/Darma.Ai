# helpers.py - PyTorch version with enhanced features

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as transforms
import matplotlib.cm as cm
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.units import inch
import io
import base64

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0)

def generate_gradcam(model, image_tensor, target_class):
    import torch.nn as nn

    gradients = []
    activations = []

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(module, input, output):
        activations.append(output)

    # ✅ Find the LAST Conv2d layer inside model.features **recursively**
    def find_last_conv(module):
        last_conv = None
        for child in module.children():
            if isinstance(child, nn.Conv2d):
                last_conv = child
            else:
                nested_conv = find_last_conv(child)
                if nested_conv is not None:
                    last_conv = nested_conv
        return last_conv

    last_conv_layer = find_last_conv(model.features)

    if last_conv_layer is None:
        raise ValueError("No Conv2d layer found in model.features.")

    handle_forward = last_conv_layer.register_forward_hook(save_activation)
    handle_backward = last_conv_layer.register_backward_hook(save_gradient)

    model.eval()
    output = model(image_tensor)
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    grads_val = gradients[0].cpu().data.numpy()[0]
    activations_val = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations_val[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)

    handle_forward.remove()
    handle_backward.remove()

    return cam

def apply_heatmap(image, heatmap):
    img = np.array(image)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cm.jet(heatmap)[:, :, :3] * 255
    heatmap = np.uint8(heatmap)
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.uint8(superimposed_img)
    return superimposed_img

def get_prediction(model, image_tensor, class_names):
    """Enhanced function that returns all class probabilities"""
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        predicted_class = class_names[predicted.item()]
        
        # Return all probabilities for bar chart
        all_probabilities = probs[0].cpu().numpy()
        
        return predicted_class, confidence.item() * 100, predicted.item(), all_probabilities

def pil_to_bytes(image):
    """Convert PIL image to bytes for PDF embedding"""
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return img_buffer

def generate_pdf_report(original_image, gradcam_image, predicted_class, confidence, description):
    """Generate PDF report with diagnosis results"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        textColor=colors.darkblue
    )
    
    story = []
    
    # Title
    story.append(Paragraph("Skin Disease Classification Report", title_style))
    story.append(Spacer(1, 20))
    
    # Diagnosis Results
    story.append(Paragraph("Diagnosis Results", styles['Heading2']))
    story.append(Paragraph(f"<b>Predicted Condition:</b> {predicted_class}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Description
    story.append(Paragraph("Medical Description", styles['Heading2']))
    story.append(Paragraph(description, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Images section
    story.append(Paragraph("Analysis Images", styles['Heading2']))
    
    try:
        # Original image
        original_buffer = pil_to_bytes(original_image)
        original_rl_img = RLImage(original_buffer, width=2*inch, height=2*inch)
        story.append(Paragraph("Original Image:", styles['Normal']))
        story.append(original_rl_img)
        story.append(Spacer(1, 10))
        
        # Grad-CAM image
        gradcam_pil = Image.fromarray(gradcam_image)
        gradcam_buffer = pil_to_bytes(gradcam_pil)
        gradcam_rl_img = RLImage(gradcam_buffer, width=2*inch, height=2*inch)
        story.append(Paragraph("AI Focus Areas (Grad-CAM):", styles['Normal']))
        story.append(gradcam_rl_img)
        
    except Exception as e:
        story.append(Paragraph(f"Note: Images could not be embedded in PDF. Error: {str(e)}", styles['Normal']))
    
    story.append(Spacer(1, 30))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.red
    )
    story.append(Paragraph("DISCLAIMER", styles['Heading3']))
    story.append(Paragraph(
        "This tool is for educational purposes only. This is not a medical diagnosis. "
        "Always consult with a qualified healthcare professional for proper medical advice.",
        disclaimer_style
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer