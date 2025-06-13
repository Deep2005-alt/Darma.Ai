import os
import csv
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define class names for cat skin diseases
class_names = ['Flea_Allergy', 'Health', 'Ringworm', 'Scabies']

class CatSkinDiseasePredictor:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.class_names = ['Flea_Allergy', 'Health', 'Ringworm', 'Scabies']

        self.model = models.resnet50(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.class_names))

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict_image(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            return {"error": f"Error opening image: {str(e)}"}

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        confidence_scores = probabilities.cpu().numpy()

        result = {
            "filename": os.path.basename(image_path),
            "probabilities": {class_name: float(confidence_scores[i]) * 100 for i, class_name in enumerate(self.class_names)},
            "prediction": self.class_names[np.argmax(confidence_scores)],
            "confidence": float(np.max(confidence_scores)) * 100
        }
        
        return result

# Disease information dictionary
disease_info = {
    "Flea_Allergy": {
        "description": "Flea allergy dermatitis (FAD) is a skin condition in cats caused by an allergic reaction to flea saliva.",
        "symptoms": "Intense itching, hair loss, skin redness, scabs, and hot spots, particularly around the base of the tail, head, neck, and thighs.",
        "treatment": "Flea control products, anti-inflammatory medications, and keeping the environment flea-free."
    },
    "Health": {
        "description": "A healthy cat skin is free from skin conditions and diseases.",
        "symptoms": "Smooth, clean coat, no excessive scratching, no visible redness, lesions, or parasites.",
        "treatment": "Regular grooming, a balanced diet, and routine veterinary check-ups to maintain skin health."
    },
    "Ringworm": {
        "description": "Ringworm is a fungal infection that affects the skin, hair, and occasionally nails of cats.",
        "symptoms": "Circular patches of hair loss, redness, scaling, and crusty skin, most commonly on the head, ears, and forelimbs.",
        "treatment": "Antifungal medications (oral and topical), environmental decontamination, and sometimes clipping the coat in long-haired cats."
    },
    "Scabies": {
        "description": "Scabies (mange) is caused by the Sarcoptes scabiei mite, which burrows into the skin causing intense irritation.",
        "symptoms": "Severe itching, redness, scaling, crusty skin lesions, hair loss, especially on the ears, face, legs, and belly.",
        "treatment": "Anti-parasitic medications, medicated baths, and environmental treatment to eliminate mites."
    }
}

# Create feedback CSV file
FEEDBACK_CSV = 'feedback_data.csv'
if not os.path.exists(FEEDBACK_CSV):
    with open(FEEDBACK_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'image_path', 'predicted_class', 'probability', 'confirmed_conditions', 'doctor_notes'])

def create_chart(probabilities):
    """Create a bar chart for prediction probabilities"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    classes = list(probabilities.keys())
    values = list(probabilities.values())
    
    y_pos = np.arange(len(classes))
    bars = ax.barh(y_pos, values, align='center', color='#c2e0b6')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlabel('Probability (%)')
    ax.set_title('Disease Probability')
    
    # Highlight the highest probability bar
    predicted_idx = np.argmax(values)
    bars[predicted_idx].set_color('#8cc084')
    
    # Add text labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_position = width + 1
        ax.text(label_position, bar.get_y() + bar.get_height()/2, 
                f'{values[i]:.1f}%', 
                va='center')
    
    # Set axis limits
    ax.set_xlim(0, 110)
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Encode the bytes buffer to base64
    chart_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    
    return chart_img

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            return render_template('index.html', error="No file part")
        
        file = request.files['image']
        
        # If user does not select a file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Initialize model and predict
            model = CatSkinDiseasePredictor('cat_skin_disease_model.pth')
            result = model.predict_image(filepath)
            
            if "error" in result:
                return render_template('index.html', error=result["error"])
            
            # Get prediction details
            predicted_class = result['prediction']
            confidence = result['confidence']
            probabilities = result['probabilities']
            
            # Create visualization
            chart_img = create_chart(probabilities)
            
            # Get disease information
            disease_description = disease_info.get(predicted_class, {}).get('description', "No information available.")
            disease_symptoms = disease_info.get(predicted_class, {}).get('symptoms', "No symptom information available.")
            disease_treatment = disease_info.get(predicted_class, {}).get('treatment', "No treatment information available.")
            
            # Pass the results to template
            return render_template('index.html', 
                                   prediction=predicted_class.replace('_', ' '),
                                   prediction_key=predicted_class,  # For checkbox identification
                                   confidence=f"{confidence:.1f}",
                                   chart_img=chart_img,
                                   image_url=filepath,
                                   description=disease_description,
                                   symptoms=disease_symptoms,
                                   treatment=disease_treatment)
                
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}")
    
    return render_template('index.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract feedback data
    user_feedback = data.get('user_feedback', {})
    confirmed_conditions = ','.join(user_feedback.get('confirmed_conditions', [])) if isinstance(user_feedback, dict) else ''
    doctor_notes = user_feedback.get('notes', '') if isinstance(user_feedback, dict) else ''
    
    # Save feedback to CSV
    with open(FEEDBACK_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            data['image_path'],
            data['predicted_class'],
            data['probability'],
            confirmed_conditions,
            doctor_notes
        ])
    
    return jsonify({'status': 'success'})

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
