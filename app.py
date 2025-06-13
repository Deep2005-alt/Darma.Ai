from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect, flash
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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
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

# Human skin condition classes
human_class_names = ['Acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Melanoma', 'Healthy']

# Human skin condition descriptions and treatments
human_descriptions = {
    'Acne': 'A common skin condition that occurs when hair follicles become clogged with oil and dead skin cells.',
    'Eczema': 'A condition that makes your skin red and itchy, often appearing as patches.',
    'Psoriasis': 'A skin disorder that causes skin cells to multiply up to 10 times faster than normal.',
    'Rosacea': 'A common skin condition that causes redness and visible blood vessels in your face.',
    'Melanoma': 'A serious form of skin cancer that begins in cells known as melanocytes.',
    'Healthy': 'No concerning skin conditions detected.'
}

human_treatments = {
    'Acne': '''
        <ul class="list-disc pl-5 space-y-2">
            <li>Topical treatments (benzoyl peroxide, salicylic acid)</li>
            <li>Antibiotics (oral or topical)</li>
            <li>Retinoids</li>
            <li>Hormonal therapy for women</li>
            <li>Isotretinoin for severe cases</li>
        </ul>
    ''',
    'Eczema': '''
        <ul class="list-disc pl-5 space-y-2">
            <li>Regular moisturizing</li>
            <li>Topical corticosteroids</li>
            <li>Antihistamines for itching</li>
            <li>Phototherapy</li>
            <li>Avoiding triggers</li>
        </ul>
    ''',
    'Psoriasis': '''
        <ul class="list-disc pl-5 space-y-2">
            <li>Topical treatments (corticosteroids, vitamin D analogues)</li>
            <li>Light therapy</li>
            <li>Systemic medications</li>
            <li>Biologic drugs</li>
            <li>Lifestyle changes</li>
        </ul>
    ''',
    'Rosacea': '''
        <ul class="list-disc pl-5 space-y-2">
            <li>Topical medications</li>
            <li>Oral antibiotics</li>
            <li>Laser therapy</li>
            <li>Trigger avoidance</li>
            <li>Gentle skin care</li>
        </ul>
    ''',
    'Melanoma': '''
        <ul class="list-disc pl-5 space-y-2">
            <li>Surgical removal</li>
            <li>Immunotherapy</li>
            <li>Targeted therapy</li>
            <li>Chemotherapy</li>
            <li>Radiation therapy</li>
        </ul>
    ''',
    'Healthy': '''
        <ul class="list-disc pl-5 space-y-2">
            <li>Maintain a regular skincare routine</li>
            <li>Use sunscreen daily</li>
            <li>Stay hydrated</li>
            <li>Eat a balanced diet</li>
            <li>Get regular exercise</li>
        </ul>
    '''
}

def load_models():
    global animal_model, human_model
    try:
        # Load animal skin disease model
        if os.path.exists('models/animal_skin_model.pth'):
            animal_model = models.resnet50(weights=None)
            num_ftrs = animal_model.fc.in_features
            animal_model.fc = nn.Linear(num_ftrs, len(dog_class_names))
            animal_model.load_state_dict(torch.load('models/animal_skin_model.pth', map_location=torch.device('cpu')))
            animal_model.eval()
            app.logger.info("Animal skin disease model loaded successfully")
        else:
            app.logger.warning("Animal skin disease model file not found")
            animal_model = None
        
        # Load human skin disease model
        if os.path.exists('models/human_skin_model.pth'):
            human_model = models.resnet50(weights=None)
            num_ftrs = human_model.fc.in_features
            human_model.fc = nn.Linear(num_ftrs, len(human_class_names))
            human_model.load_state_dict(torch.load('models/human_skin_model.pth', map_location=torch.device('cpu')))
            human_model.eval()
            app.logger.info("Human skin disease model loaded successfully")
        else:
            app.logger.warning("Human skin disease model file not found")
            human_model = None
            
    except Exception as e:
        app.logger.error(f"Error loading models: {str(e)}")
        animal_model = None
        human_model = None

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
def dog_index():
    if animal_model is None:
        return render_template('DogDerma/index.html', error="Model not available. Please contact the administrator.")
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('DogDerma/index.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('DogDerma/index.html', error="No file selected")
        
        if file:
            try:
                # Save the uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process the image
                image = Image.open(filepath).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image_tensor = transform(image).unsqueeze(0)
                
                # Get prediction
                with torch.no_grad():
                    outputs = animal_model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                # Get class name and description
                predicted_class_name = dog_class_names[predicted_class]
                description = dog_disease_info.get(predicted_class_name, {}).get('description', "No description available")
                treatment = dog_disease_info.get(predicted_class_name, {}).get('treatment', "No treatment information available")
                
                # Create probability distribution chart
                plt.figure(figsize=(10, 6))
                plt.bar(dog_class_names, probabilities[0].numpy())
                plt.xticks(rotation=45, ha='right')
                plt.title('Probability Distribution')
                plt.tight_layout()
                
                # Save the chart
                chart_path = os.path.join(app.config['UPLOAD_FOLDER'], 'probability_chart.png')
                plt.savefig(chart_path)
                plt.close()
                
                return render_template('DogDerma/index.html',
                                     filename=filename,
                                     prediction=predicted_class_name,
                                     confidence=confidence,
                                     description=description,
                                     treatment=treatment,
                                     chart_path='probability_chart.png')
                
            except Exception as e:
                app.logger.error(f"Error processing image: {str(e)}")
                return render_template('DogDerma/index.html', error="Error processing image")
    
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

@app.route('/humanderma')
def human_home():
    return render_template('HumanSkinDisease/home.html')

@app.route('/humanderma/index', methods=['GET', 'POST'])
def human_index():
    if human_model is None:
        return render_template('HumanSkinDisease/index.html', error="Model not available. Please contact the administrator.")
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('HumanSkinDisease/index.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('HumanSkinDisease/index.html', error="No file selected")
        
        if file:
            try:
                # Save the uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process the image
                image = Image.open(filepath).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image_tensor = transform(image).unsqueeze(0)
                
                # Get prediction
                with torch.no_grad():
                    outputs = human_model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                # Get class name and description
                predicted_class_name = human_class_names[predicted_class]
                description = human_descriptions.get(predicted_class_name, "No description available")
                treatment = human_treatments.get(predicted_class_name, "No treatment information available")
                
                # Create probability distribution chart
                plt.figure(figsize=(10, 6))
                plt.bar(human_class_names, probabilities[0].numpy())
                plt.xticks(rotation=45, ha='right')
                plt.title('Probability Distribution')
                plt.tight_layout()
                
                # Save the chart
                chart_path = os.path.join(app.config['UPLOAD_FOLDER'], 'probability_chart.png')
                plt.savefig(chart_path)
                plt.close()
                
                return render_template('HumanSkinDisease/index.html',
                                     filename=filename,
                                     prediction=predicted_class_name,
                                     confidence=confidence,
                                     description=description,
                                     treatment=treatment,
                                     chart_path='probability_chart.png')
                
            except Exception as e:
                app.logger.error(f"Error processing image: {str(e)}")
                return render_template('HumanSkinDisease/index.html', error="Error processing image")
    
    return render_template('HumanSkinDisease/index.html')

@app.route('/humanderma/about')
def human_about():
    return render_template('HumanSkinDisease/about.html')

@app.route('/humanderma/feedback', methods=['POST'])
def human_feedback():
    # Handle doctor's feedback
    doctor_name = request.form.get('doctor_name')
    patient_age = request.form.get('patient_age')
    conditions = request.form.getlist('conditions')
    notes = request.form.get('notes')
    
    # Here you would typically save the feedback to a database
    # For now, we'll just flash a success message
    flash('Thank you for your feedback!', 'success')
    return redirect(url_for('human_index'))

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise 