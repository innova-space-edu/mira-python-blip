# -*- coding: utf-8 -*-
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Carga perezosa para acelerar arranque en Render
_processor_blip = None
_model_blip = None
_clip_model = None
_clip_processor = None

def _lazy_load():
    global _processor_blip, _model_blip, _clip_model, _clip_processor
    if _processor_blip is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
        _processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

from utils import decode_base64_image

app = Flask(__name__)
CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "*").split(","))


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"}), 200


@app.route("/blip", methods=["POST"])
def blip():
    """
    Body: { "image": "data:image/...;base64,...", "prompt": "opcional" }
    """
    _lazy_load()

    data = request.get_json(force=True, silent=True) or {}
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "Debes enviar 'image' (base64)"}), 400

    try:
        img = decode_base64_image(img_b64)

        # Descripción con BLIP
        inputs = _processor_blip(img, return_tensors="pt")
        out = _model_blip.generate(**inputs, max_new_tokens=60)
        description = _processor_blip.decode(out[0], skip_special_tokens=True)

        # Etiquetado con CLIP (top 3 de un conjunto fijo de etiquetas)
        labels = [
            "classroom", "student", "teacher", "computer", "desk",
            "graph", "board", "notebook", "experiment", "person",
            "table", "plant", "science", "book", "screen"
        ]
        clip_inputs = _clip_processor(text=labels, images=img, return_tensors="pt", padding=True)
        clip_outputs = _clip_model(**clip_inputs)
        probs = clip_outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
        top = probs.argsort()[-3:][::-1]
        tags = [labels[i] for i in top]

        scene = "interior" if any(t in tags for t in ["classroom", "desk", "table", "computer"]) else "exterior"

        return jsonify({
            "description": description,
            "tags": tags,
            "scene": scene
        })
    except Exception as e:
        return jsonify({"error": f"Fallo BLIP/CLIP: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=True)
