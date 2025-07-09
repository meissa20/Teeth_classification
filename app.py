import streamlit as st
import torch
from torchvision import models, transforms
import json
from PIL import Image

def evaluate(model, image_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        with open('label_encoder.json', 'r') as f:
            label_encoder = json.load(f)
            idx_to_class = {int(v): k for k, v in label_encoder.items()}  # Convert str→int keys
        predicted_class = idx_to_class[preds.item()]  # Use .item() to get int from tensor
    return predicted_class

device = torch.device("cpu")
model = models.densenet169(pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, 7)
model.load_state_dict(torch.load("best_model_transfer.pth", map_location=device))
model.to(device)

st.title("Please enter your image!")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

normal_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = normal_transform(image).unsqueeze(0).to(device)
    predicted_class = evaluate(model, input_tensor)
    st.success(f"Predicted class: **{predicted_class}**")

