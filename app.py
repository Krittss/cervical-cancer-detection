from flask import Flask, render_template, request, send_from_directory

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2

from gradcam import make_gradcam_heatmap, overlay_heatmap

app = Flask(__name__)

# ✅ Load trained model
model = tf.keras.models.load_model("model.h5")

# ✅ Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ✅ SAME CLASS ORDER AS TRAINING (VERY IMPORTANT)
class_names = [
    "Cancer",
    "light_dysplastic",
    "moderate_dysplastic",
    "normal_columnar",
    "normal_intermediate",
    "normal_sperficial",
    "severe_dysplastic"
]

# ✅ Label mapping (for UI)
label_map = {
    "Cancer": "Cancer",
    "light_dysplastic": "Mild Dysplasia",
    "moderate_dysplastic": "Moderate Dysplasia",
    "severe_dysplastic": "Severe Dysplasia",
    "normal_columnar": "Normal Columnar",
    "normal_intermediate": "Normal Intermediate",
    "normal_sperficial": "Normal Superficial"
}

# ✅ Risk mapping
risk_map = {
    "Normal Superficial": "Low Risk",
    "Normal Intermediate": "Low Risk",
    "Normal Columnar": "Low Risk",
    "Mild Dysplasia": "Medium Risk",
    "Moderate Dysplasia": "High Risk",
    "Severe Dysplasia": "Critical",
    "Cancer": "Critical"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result")
def result():
    return render_template(
        "result.html",
        prediction=None,
        confidence=None,
        risk=None,
        heatmap=None,
        image=None
    )

@app.route("/screening")
def screening():
    return render_template("screening.html")

@app.route("/about")
def about():
    return render_template("about.html")

# ✅ MAIN PREDICTION ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    if file.filename == "":
        return "No file uploaded"

    # Save uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # ✅ PREPROCESSING (MATCH TRAINING EXACTLY)
    img = Image.open(filepath).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)

    # 🔥 CRITICAL FIX: SAME AS TRAINING
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Prediction
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred) * 100)

    raw_label = class_names[class_idx]
    label = label_map[raw_label]
    risk = risk_map[label]

    print("\nPrediction vector:", pred)
    print("Raw class:", raw_label)
    print("Final label:", label)
    print("Confidence:", confidence)

    # ✅ Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model)

    img_cv = cv2.imread(filepath)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # FIX COLOR
    img_cv = cv2.resize(img_cv, (224, 224))

    cam = overlay_heatmap(heatmap, img_cv)

    heatmap_path = "static/gradcam.jpg"
    cv2.imwrite(heatmap_path, cam)

    # Image URL for frontend
    image_url = f"/uploads/{file.filename}"

    return render_template(
        "result.html",
        prediction=label,
        confidence=round(confidence, 2),
        risk=risk,
        heatmap="/" + heatmap_path,
        image=image_url
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)