import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Define the path to save processed images
results_path = r"C:\Users\Nithin Maloth\Desktop\processed images"

# Specify the desired target size for resizing
target_size = (224, 224)

# Define class names
classes = ['glioma', 'meningioma', 'no tumor', 'pituitary']

# Load the trained model
model_path = os.path.join(results_path, "model_name.h5")
loaded_model = load_model(model_path)

# Path to the new image you want to classify
new_image_path = r"C:\Users\Nithin Maloth\Desktop\brain scan\Testing\notumor\Te-no_0367.jpg"

# Preprocess the new image
new_image = image.load_img(new_image_path, target_size=target_size)
new_image_array = image.img_to_array(new_image)
new_image_array = np.expand_dims(new_image_array, axis=0)
new_image_array /= 255.0

# Predict class probabilities
class_probabilities = loaded_model.predict(new_image_array)

# Interpret the results
predicted_class_index = np.argmax(class_probabilities)
predicted_class = classes[predicted_class_index]

# Set a threshold for anomaly detection
anomaly_threshold = 0.5  # You can adjust this threshold as needed

# Determine if it's an anomaly based on the threshold
is_anomaly = np.max(class_probabilities) < anomaly_threshold

# Visualize the new image and results
plt.imshow(new_image)
plt.title(f"Predicted Class: {predicted_class}\nProbabilities: {class_probabilities}")
if is_anomaly:
    plt.xlabel("Anomaly Detected")
plt.show()
