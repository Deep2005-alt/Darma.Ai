from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import csv
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static')

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load human skin disease model
HUMAN_MODEL_PATH = 'models/human_skin_model.pth'
# Load animal skin disease model
ANIMAL_MODEL_PATH = 'models/animal_skin_model.pth'

# Initialize models
human_model = None
animal_model = None

# Dog skin disease classes
dog_class_names = [
    'Dermatitis',
    'Fungal_infections',
    'Healthy',
    'Hypersensitivity',
    'demodicosis',
    'ringworm'
]

# Dog disease information
dog_disease_info = {
    'Dermatitis': {
        'description': "Dermatitis in dogs is a general term for skin inflammation, leading to redness, itching, and rash. Common causes include allergies (atopy, food), irritants, and secondary bacterial or yeast infections.",
        'treatment': "Treatment often involves identifying and addressing the underlying cause, along with medications like corticosteroids or antihistamines to manage itching and inflammation. Special shampoos and topical treatments are also frequently used. Treating secondary infections is crucial."
    },
    'Fungal_infections': {
        'description': "Fungal infections in dogs can cause a variety of skin issues, with ringworm and yeast dermatitis (Malassezia) being the most common. Symptoms include itching, redness, hair loss, scaling, and a characteristic circular lesion in the case of ringworm. Yeast infections often occur in skin folds.",
        'treatment': "Treatment depends on the specific fungus but typically involves antifungal medications, either topical (creams, shampoos) or oral. Environmental decontamination is important for ringworm."
    },
    'Healthy': {
        'description': "Healthy dog skin is characterized by a soft, pliable texture, a pink or pigmented color (depending on the breed), and a coat that is shiny and free of excessive shedding. There should be no signs of itching, redness, or irritation.",
        'treatment': "Maintaining healthy skin in dogs involves a balanced diet rich in omega-3 fatty acids, regular grooming, appropriate bathing, parasite control, and avoiding allergens and irritants."
    },
    'Hypersensitivity': {
        'description': "Hypersensitivity in dogs refers to allergic reactions of the skin. Common manifestations include atopic dermatitis (environmental allergies), food allergies, and contact allergies. Symptoms include intense itching, scratching, rubbing, chewing, leading to redness, hair loss, secondary infections, and skin lesions.",
        'treatment': "Treatment involves identifying and avoiding the allergen (if possible), managing itching with antihistamines, corticosteroids, or other immunomodulatory drugs (like Apoquel or Cytopoint), and treating secondary infections. Food trials are used for food allergies. Allergy testing and immunotherapy (allergy shots) may be helpful for atopy."
    },
    'demodicosis': {
        'description': "Demodicosis in dogs is caused by Demodex mites. There are two main forms: localized (often seen in puppies, with small, patchy areas of hair loss) and generalized (more severe, potentially indicating an underlying immune deficiency). Symptoms include hair loss, scaling, redness, and secondary bacterial infections.",
        'treatment': "Treatment depends on the form. Localized demodicosis may resolve on its own. Generalized demodicosis requires aggressive treatment with miticidal medications (like oral ivermectin, milbemycin oxime, or topical amitraz), often combined with antibiotics for secondary infections. Underlying health issues need to be addressed."
    },
    'ringworm': {
        'description': "Ringworm in dogs is a fungal infection that affects the skin and hair. It's zoonotic, meaning it can be transmitted to humans. Lesions are often circular, with hair loss, scaling, and redness. It can be itchy, though not always severely.",
        'treatment': "Treatment involves antifungal medications, either topical or oral. Environmental decontamination is crucial to prevent spread. Clipping the hair around lesions can help with topical treatment. Treatment duration is typically prolonged (several weeks to months)."
    }
}

def load_models():
    global human_model, animal_model
    try:
        if os.path.exists(HUMAN_MODEL_PATH):
            logger.info("Loading human skin disease model...")
            # Initialize the human model architecture first
            human_model = models.resnet50(weights=None)
            num_ftrs = human_model.fc.in_features
            human_model.fc = nn.Linear(num_ftrs, 7)  # Assuming 7 classes for human skin diseases
            
            # Load the state dictionary
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            human_model.load_state_dict(torch.load(HUMAN_MODEL_PATH, map_location=device))
            human_model.to(device)
            human_model.eval()
            logger.info("Human skin disease model loaded successfully")
        
        if os.path.exists(ANIMAL_MODEL_PATH):
            logger.info("Loading animal skin disease model...")
            # Load the dog skin disease model
            animal_model = models.resnet50(weights=None)
            num_ftrs = animal_model.fc.in_features
            animal_model.fc = nn.Linear(num_ftrs, len(dog_class_names))
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")
            
            animal_model.load_state_dict(torch.load(ANIMAL_MODEL_PATH, map_location=device))
            animal_model.to(device)
            animal_model.eval()
            logger.info("Animal skin disease model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# Load models when app starts
load_models()

def create_dog_chart(probabilities):
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        y_pos = np.arange(len(dog_class_names))
        
        bars = ax.barh(y_pos, probabilities * 100, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(dog_class_names)
        ax.invert_yaxis()
        ax.set_xlabel('Probability (%)')
        ax.set_title('Disease Probability')
        
        predicted_idx = np.argmax(probabilities)
        bars[predicted_idx].set_color('#4CAF50')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_position = width + 1
            ax.text(label_position, bar.get_y() + bar.get_height()/2, 
                    f'{probabilities[i]*100:.1f}%', 
                    va='center')
        
        ax.set_xlim(0, 110)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        chart_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)
        
        return chart_img
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/predictions')
def predictions():
    return render_template('predictions.html')

@app.route('/predict/human', methods=['POST'])
def predict_human():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    # Add your human skin disease prediction logic here
    # This is a placeholder response
    return jsonify({
        'patient_id': 'HUMAN_123',
        'predictions': {
            'Disease A': 75.5,
            'Disease B': 15.2,
            'Disease C': 9.3
        }
    })

@app.route('/predict/animal', methods=['POST'])
def predict_animal():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Image saved to {filepath}")

        # Open and preprocess the image
        image = Image.open(filepath).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)
        
        logger.info("Image preprocessed successfully")
        
        # Make prediction
        with torch.no_grad():
            outputs = animal_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
            
            predicted_idx = np.argmax(probabilities)
            predicted_class = dog_class_names[predicted_idx]
            confidence = float(probabilities[predicted_idx] * 100)
            
            logger.info(f"Prediction made: {predicted_class} with {confidence:.2f}% confidence")
            
            chart_img = create_dog_chart(probabilities)
            
            disease_description = dog_disease_info.get(predicted_class, {}).get('description', "No information available.")
            disease_treatment = dog_disease_info.get(predicted_class, {}).get('treatment', "No treatment information available.")
            
            # Convert all probabilities to native Python float
            predictions_dict = {k: float(v * 100) for k, v in zip(dog_class_names, probabilities)}
            
            return jsonify({
                'patient_id': 'DOG_123',
                'predictions': predictions_dict,
                'chart_img': chart_img,
                'description': disease_description,
                'treatment': disease_treatment
            })
            
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        doctor_name = request.form.get('doctor_name', '').strip()
        dog_age = request.form.get('dog_age', '').strip()
        confirmed_conditions = request.form.getlist('conditions')
        doctor_notes = request.form.get('doctor_notes', '').strip()
        image_path = request.form.get('image_path', '')
        predicted_class = request.form.get('predicted_class', '')
        probability = request.form.get('probability', '')

        if not doctor_name or not dog_age:
            return jsonify({'error': "Doctor's name and dog age are required."}), 400

        feedback_file = 'feedback_data.csv'
        if not os.path.exists(feedback_file):
            with open(feedback_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'image_path', 'predicted_class', 'probability', 'confirmed_conditions', 'doctor_notes', 'doctor_name', 'dog_age'])

        with open(feedback_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                image_path,
                predicted_class,
                probability,
                ','.join(confirmed_conditions),
                doctor_notes,
                doctor_name,
                dog_age
            ])

        logger.info(f"Feedback submitted successfully for image: {image_path}")
        return jsonify({'message': 'Feedback submitted successfully'})
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/dogderma')
def dogderma():
    return render_template('DogDerma/home.html')

@app.route('/dogderma/index', methods=['GET', 'POST'])
def dogderma_index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('DogDerma/index.html', error="No file part")
        
        file = request.files['image']
        
        if file.filename == '':
            return render_template('DogDerma/index.html', error="No selected file")
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            image = Image.open(filepath).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            image_tensor = image_tensor.to(device)
            
            with torch.no_grad():
                outputs = animal_model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
                
                predicted_idx = np.argmax(probabilities)
                predicted_class = dog_class_names[predicted_idx]
                confidence = probabilities[predicted_idx] * 100
                
                chart_img = create_dog_chart(probabilities)
                
                disease_description = dog_disease_info.get(predicted_class, {}).get('description', "No information available.")
                disease_treatment = dog_disease_info.get(predicted_class, {}).get('treatment', "No treatment information available.")
                
                return render_template('DogDerma/index.html', 
                                      prediction=predicted_class,
                                      confidence=f"{confidence:.1f}",
                                      chart_img=chart_img,
                                      image_url=f"/static/uploads/{filename}",
                                      description=disease_description,
                                      treatment=disease_treatment)
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return render_template('DogDerma/index.html', error=f"An error occurred: {str(e)}")
    
    return render_template('DogDerma/index.html')

@app.route('/dogderma/about')
def dogderma_about():
    return render_template('DogDerma/about.html')

@app.route('/dogderma/submit_feedback', methods=['POST'])
def dogderma_submit_feedback():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    doctor_name = request.form.get('doctor_name', '').strip()
    dog_age = request.form.get('dog_age', '').strip()
    confirmed_conditions = request.form.getlist('conditions')
    doctor_notes = request.form.get('doctor_notes', '').strip()
    image_path = request.form.get('image_path', '')
    predicted_class = request.form.get('predicted_class', '')
    probability = request.form.get('probability', '')

    if not doctor_name or not dog_age:
        return render_template('DogDerma/index.html', error="Doctor's name and dog age are required.")

    with open('feedback_data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            image_path,
            predicted_class,
            probability,
            ','.join(confirmed_conditions),
            doctor_notes,
            doctor_name,
            dog_age
        ])

    return render_template('DogDerma/feedback_submitted.html', doctor_name=doctor_name, dog_age=dog_age)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise 