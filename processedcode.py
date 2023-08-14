import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Define the parent path containing the Training and Testing folders
parent_dir = r"C:\Users\Nithin Maloth\Desktop\brain scan"

# Define the path to save processed images
results_path = r"C:\Users\Nithin Maloth\Desktop\processed images"

# Specify the desired target size for resizing
target_size = (224, 224)

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# No augmentation for test data
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Create data generators
batch_size = 32
classes = ['glioma', 'meningioma', 'no tumor', 'pituitary']

train_generator = train_datagen.flow_from_directory(
    os.path.join(parent_dir, 'Training'),
    target_size=target_size,
    batch_size=batch_size,
    classes=classes,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(parent_dir, 'Testing'),
    target_size=target_size,
    batch_size=batch_size,
    classes=classes,
    class_mode='categorical'
)

# Define your model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(12, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(12, activation='relu'),
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(
    train_generator,
    epochs=20,  # Adjust as needed
    validation_data=test_generator,
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_generator)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save the trained model
model.save(os.path.join(results_path, "model_name.h5"))
