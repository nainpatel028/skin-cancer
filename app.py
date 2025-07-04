import os
import torch
from flask import Flask, request, render_template, jsonify
from torchvision import models, transforms
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.vgg19(weights=None)  # Specify `weights=None` to avoid warnings
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)

# Load the model weights correctly
checkpoint_path = os.path.join(os.getcwd(), "models", "model_epoch_50.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load state_dict into the model
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for VGG
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels
int_to_label = {0: "Non-Melanoma", 1: "Melanoma"}

# Homepage route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Preprocess the image
        image = Image.open(file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            prediction = int_to_label[predicted.item()]

        return render_template("index.html", prediction=prediction)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
