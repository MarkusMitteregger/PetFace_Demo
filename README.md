# 🐾 Pet Identification Demo

This repository provides a **Streamlit-based demo** that identifies pets using a pretrained **Siamese (triplet-loss) model**.  
The system computes embeddings (feature vectors) from uploaded or captured images and compares them to registered pets to determine the most likely match.

---

## 📁 Project Structure
├── demo_app.py # Streamlit app for user interaction (upload / camera / identification)
├── pet_identifier.py # Core logic: model loading, preprocessing, embeddings & comparison
├── requirements.txt # Dependencies for local and cloud deployment
├── best_efficientnet_triplet.pth # Pretrained Siamese model 

## ⚙️ Setup and Installation (Local)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/PetIdentifierDemo.git
cd PetIdentifierDemo
```

## Create Invironment:

python -m venv venv

source venv/bin/activate       # macOS / Linux

venv\Scripts\activate          # Windows
### Dependencies
pip install -r requirements.txt

## 🚀 Running the App Locally
streamlit run demo_app.py

#### Open Browser (e.g):
Local URL: http://localhost:8501


## ☁️ Running in Streamlit Cloud
Link: https://petfacedeployment.streamlit.app/
