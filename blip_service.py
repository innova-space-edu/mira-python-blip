from flask import Flask, request, jsonify
from utils import decode_base64_image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel

app = Flask(__name__)

# BLIP para descripción
processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# CLIP para etiquetas/objetos
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

@app.route('/blip', methods=['POST'])
def blip():
    data = request.get_json()
    img_b64 = data.get('image', '')
    prompt = data.get('prompt', '')
    img = decode_base64_image(img_b64)

    # Descripción con BLIP
    inputs = processor_blip(img, return_tensors="pt")
    out = model_blip.generate(**inputs)
    description = processor_blip.decode(out[0], skip_special_tokens=True)

    # Etiquetado de objetos con CLIP (top 5 labels)
    labels = ["classroom", "student", "teacher", "computer", "desk", "graph", "board", "notebook", "experiment", "person", "table", "plant", "science"]
    clip_inputs = clip_processor(text=labels, images=img, return_tensors="pt", padding=True)
    clip_outputs = clip_model(**clip_inputs)
    logits_per_image = clip_outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).detach().numpy()[0]
    tags = [labels[i] for i in probs.argsort()[-3:][::-1]]

    # Mock de clasificación de escena
    scene = "interior" if "classroom" in tags or "desk" in tags else "exterior"

    return jsonify({
        "description": description,
        "tags": tags,
        "scene": scene
    })

if __name__ == "__main__":
    app.run(port=5002, debug=True)
