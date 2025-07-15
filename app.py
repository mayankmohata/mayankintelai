import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import joblib

# -------- Model Class --------
class HybridClinicalModel(nn.Module):
    def __init__(self, tabular_dim, n_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, input_ids, attention_mask, tabular_input):
        text_feat = self.bert(input_ids, attention_mask).last_hidden_state.mean(dim=1)
        tab_feat = self.tabular_mlp(tabular_input)
        combined = torch.cat((text_feat, tab_feat), dim=1)
        return self.classifier(combined)

# -------- Load Model and Tools --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HybridClinicalModel(tabular_dim=14, n_classes=3).to(device)
model.load_state_dict(torch.load("triage_model1.pt", map_location=device))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
label_enc = joblib.load("label_encoder1.pkl")
ohe = joblib.load("ohe_encoder1.pkl")

# -------- Streamlit UI --------
st.title("🩺 Medical Triage AI Assistant")

symptoms = st.text_area("Symptoms", "Shortness of breath and chest pain")
age = st.number_input("Age", 0, 100, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
history = st.selectbox("Past Medical History", ["None", "Hypertension", "Diabetes", "Asthma", "Heart Disease", "Anxiety Disorder", "Tuberculosis", "Epilepsy"])
hr = st.slider("Heart Rate (bpm)", 50, 150, 90)
temp = st.slider("Temperature (°C)", 35.0, 41.0, 37.0)
spo2 = st.slider("SpO2 (%)", 85, 100, 96)

if st.button("Predict Urgency Level"):
    # Prepare tabular input
    tab_input = np.hstack([[[age, hr, temp, spo2]], ohe.transform([[gender, history]]).toarray()])

    tab_tensor = torch.tensor(tab_input, dtype=torch.float32).to(device)

    # Tokenize symptom text
    enc = tokenizer(symptoms, return_tensors="pt", truncation=True, padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Predict
    with torch.no_grad():
        output = model(input_ids, attention_mask, tab_tensor)
        pred = torch.argmax(output, dim=1).item()
        result = label_enc.inverse_transform([pred])[0]

    st.success(f"🚨 Predicted Urgency: **{result.upper()}**")
