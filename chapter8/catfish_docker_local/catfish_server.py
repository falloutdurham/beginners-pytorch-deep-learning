import os
import requests
import torch
from flask import Flask, jsonify, request
from io import BytesIO
from PIL import Image
from torchvision import transforms

from catfish_model import CatfishModel, CatfishClasses


def load_model():
    location = os.environ["CATFISH_MODEL_LOCATION"]
    m = CatfishModel
    m.load_state_dict(torch.load(location, map_location="cpu"))
    m.eval()
    return m


model = load_model()

img_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
])

def create_app():
    app = Flask(__name__)

    @app.route("/")
    def status():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=['GET', 'POST'])
    def predict():
        if request.method == 'POST':
            img_url = request.form.image_url
        else:
            img_url = request.args.get('image_url', '')

        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        img_tensor = img_transforms(img).unsqueeze(0)
        prediction =  model(img_tensor)
        predicted_class = CatfishClasses[torch.argmax(prediction)]
        return jsonify({"image": img_url, "prediction": predicted_class})

    return app