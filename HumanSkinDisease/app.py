import os
import csv
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import transforms, models

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define number of classes for your model
num_classes = 23  # Changed from 7 to 23 to match the saved model

# Use ResNet model
model = models.resnet18(pretrained=False)
# Replace the final fc layer with the custom structure from your saved model
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 512),
    torch.nn.BatchNorm1d(512),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(512, num_classes)
)

# Load the state dictionary
state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# Define image transformations (adjust according to your model's input)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Update the class names with actual disease names
class_names = {
    0: "Acne and Rosacea",
    1: "Actinic Keratosis Basal Cell Carcinoma",
    2: "Atopic Dermatitis",
    3: "Bullous Disease",
    4: "Cellulitis Impetigo",
    5: "Eczema",
    6: "Exanthems and Drug Eruptions",
    7: "Hair Loss Photos Alopecia and other Hair Diseases",
    8: "Herpes HPV and other STDs",
    9: "Light Diseases and Disorders of Pigmentation",
    10: "Lupus and other Connective Tissue diseases",
    11: "Melanoma Skin Cancer Nevi and Moles",
    12: "Nail Fungus and other Nail Disease",
    13: "Poison Ivy Photos and other Contact Dermatitis",
    14: "Psoriasis pictures Lichen Planus and related diseases",
    15: "Scabies Lyme Disease and other Infestations and Bites",
    16: "Seborrheic Keratoses and other Benign Tumors",
    17: "Systemic Disease",
    18: "Tinea Ringworm Candidiasis and other Fungal Infections",
    19: "Urticaria Hives",
    20: "Vascular Tumors",
    21: "Vasculitis Photos",
    22: "Warts Molluscum and other Viral Infections"
}

# Add this after your existing configurations
FEEDBACK_CSV = 'feedback_data.csv'

# Create CSV file with headers if it doesn't exist
if not os.path.exists(FEEDBACK_CSV):
    with open(FEEDBACK_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'image_path', 'predicted_class', 'probability', 'user_feedback'])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Save the uploaded file
        file = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and preprocess the image
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_prob, top5_classes = torch.topk(probabilities, 5)
            
            # Create list of top 5 predictions with their probabilities
            predictions = []
            for i in range(5):
                predictions.append({
                    'class': class_names[top5_classes[i].item()],
                    'probability': float(top5_prob[i].item() * 100),
                    'class_id': top5_classes[i].item()  # Add class ID for feedback
                })

        return render_template('index.html', predictions=predictions, image_url=filepath)
    
    return render_template('index.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save feedback to CSV
    with open(FEEDBACK_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            data['image_path'],
            data['predicted_class'],
            data['probability'],
            data['user_feedback']
        ])
    
    return jsonify({'status': 'success'})

# Add these routes for other pages
@app.route('/view-predictions')
def view_predictions():
    return render_template('view-predictions.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
