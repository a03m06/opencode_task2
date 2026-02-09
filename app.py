import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# -------------------------
# Simple Neural Network
# -------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    model = SimpleNN()
    state_dict = torch.load(
        "simplenn_cifar10.pth",
        map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()

# -------------------------
# UI
# -------------------------
st.title("CIFAR-10 Classification (Neural Network from Scratch)")
st.write("Simple Fully Connected Neural Network")

classes = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    st.success(f"Prediction: **{classes[predicted.item()]}**")
