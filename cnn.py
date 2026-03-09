"""
Datasets used:
- Car images: https://www.kaggle.com/datasets/kshitij192/cars-image-dataset
- No_car category images: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
- For car prediction, run the script with: python cnn.py --predict path_to_image.jpg or --prediction path_to_image.jpg
- For validation, are used images from the internet that are not in the training and test datasets, to ensure the model's generalization capability.
- The model is trained for 50 epochs, which should provide better accuracy. Adjust the number of epochs as needed based on your dataset size and performance requirements.
"""
 
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Use GPU (CUDA) if available, else fallback to CPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU (CUDA)")
    except Exception as e:
        print(f"Could not set GPU memory growth: {e}\nFalling back to CPU.")
else:
    print("No GPU found, using CPU.")

IMAGE_SIZE = 128

# Data generators/augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode='sparse',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode='sparse',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
print('Classes:', class_names)

# CNN model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(class_names), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set number of epochs
EPOCHS = 50  # Increased for better training if needed based on the dataset size and performance requirements

def main():
    if len(sys.argv) > 2 and sys.argv[1] in ['--predict', '--prediction']:
        predict_image(sys.argv[2])
    else:
        # Train the model
        history = model.fit(
            train_generator,
            epochs=EPOCHS
        )
        # Save the model
        model.save('car_cnn_model.h5')

        # Evaluate on test data
        test_loss, test_acc = model.evaluate(test_generator)
        print(f"Test accuracy: {test_acc:.4f}")

        # Plot training accuracy values
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.title('Model Training Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

def predict_image(image_path):
    from tensorflow.keras.models import load_model

    # Load the saved model
    model = load_model('car_cnn_model.h5')

    # Load and preprocess the image
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    predicted_label = class_names[predicted_class]
    confidence_percent = confidence * 100
    # Provide feedback based on confidence level
    if confidence_percent > 80:
            print(f"Predicted class: {predicted_label}")
            print(f"Prediction confidence: {confidence_percent:.2f}%")
    elif 40 < confidence_percent <= 80:
            print(f"Low confidence, might be: {predicted_label}")
            print(f"Prediction confidence: {confidence_percent:.2f}%")
    else:
            print("Prediction: Not a car or bad image quality.")
            print(f"Prediction confidence: {confidence_percent:.2f}%")

if __name__ == "__main__":
    main()