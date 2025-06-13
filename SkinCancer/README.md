# README.md - Updated for PyTorch version

## Skin Disease Classification System (PyTorch Version)
This project creates a web application that identifies skin diseases from uploaded photos and provides visual and textual explanations using a deep learning model.

---

## 🔧 Setup Instructions

### 1. Install Required Libraries
```bash
pip install torch torchvision pillow streamlit numpy matplotlib opencv-python
```

### 2. Dataset Preparation
Download and extract the HAM10000 dataset from:
[https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

Organize your dataset as follows:
```
skin_dataset/
  ├── actinic_keratosis/
  │   ├── image1.jpg
  │   ├── image2.jpg
  ├── basal_cell_carcinoma/
  ├── benign_keratosis/
  └── ...
```

### 3. Train the Model
Run the training script:
```bash
python train_model.py
```
This will save the model to:
```
skin_disease_model.pth
```

### 4. Run the Web Application
Start the Streamlit app:
```bash
streamlit run app.py
```
Then open your browser to:
```
http://localhost:8501
```

### 5. Using the Application
- Upload a skin image
- Wait for AI analysis
- Review the predicted disease and confidence
- View Grad-CAM heatmap highlighting important image regions

---

## ⚠️ Important Notes
- This application is for educational purposes only
- Always consult a healthcare professional for medical advice
- Accuracy depends on the quality and diversity of training data
- Images should be well-lit and focused for best results

---

## 🔁 Customization
- To add new classes: update `CLASSES` and `DESCRIPTIONS` in `app.py` and `helpers.py`
- To improve accuracy: tune hyperparameters in `train_model.py`
- For UI changes: edit the Streamlit layout in `app.py`
