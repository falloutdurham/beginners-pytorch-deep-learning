from flask import Flask, jsonify, request
from torchvision import transforms
import torch
import os

def load_model():
  return model

#catfish_model = load_model() 

app = Flask(__name__)

@app.route("/")
def status():
  return jsonify({"status": "ok"})

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img_url = request.form.image_url
    else:
        img_url = request.args.get('image_url','')
   
  img_url = request.image_url
  img_tensor = open_image(BytesIO(response.content))
  prediction = model(img_tensor)
  predicted_class = CatfishClasses[torch.argmax(prediction)]
  return jsonify({"image": img_url, "prediction": predicted_class})


if __name__ == '__main__':
  app.run(host=os.environ["CATFISH_HOST"], port=os.environ["CATFISH_PORT"])
