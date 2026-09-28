from pathlib import Path
import json
import sys

import numpy as np
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================



PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "tomato_mobilenetv2_v2_93_43.keras"

CLASS_NAMES_PATH = PROJECT_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Load image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    # Convert to array
    image_array = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Model prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Highest probability class
    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    return predicted_class, confidence


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python src/predict.py <image_path>"
        )

        print(
            "\nExample:"
        )

        print(
            "python src/predict.py data/demo/leaf.jpg"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    try:

        disease, confidence = predict_image(
            image_path
        )

        print("\n" + "=" * 60)
        print("TOMATO DISEASE PREDICTION")
        print("=" * 60)

        print(
            f"\nPrediction : {disease}"
        )

        print(
            f"Confidence : {confidence * 100:.2f}%"
        )

        print()

    except Exception as e:

        print(
            f"\nERROR: {e}"
        )

        sys.exit(1)

