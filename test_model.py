import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

IMG_SIZE = 160

# 🔹 Load model
model = load_model("models/leaf_disease_model.h5")

# 🔹 Final validation accuracy from training
MODEL_ACCURACY = 0.8538   # 85.38%

class_names = [
    "diseased_moringa",
    "diseased_papaya",
    "healthy_moringa",
    "healthy_papaya"
]

# 🔹 Image path
img_path = "test_img.jpg"

# 🔹 Preprocess image
img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 🔹 Measure prediction time
start_time = time.time()

pred = model.predict(img_array)

end_time = time.time()
prediction_time = end_time - start_time

# 🔹 Get results
predicted_class = class_names[np.argmax(pred)]
confidence = np.max(pred)

# 🔹 Print output
print("\n Prediction:", predicted_class)
print(" Confidence: {:.2f}%".format(confidence * 100))
print(" Model Accuracy: {:.2f}%".format(MODEL_ACCURACY * 100))
print("⏱ Prediction Time: {:.4f} seconds".format(prediction_time))
