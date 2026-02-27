# Car-Brand-Recognition-CNN

Convolutional Neural Network (CNN) built with TensorFlow/Keras for detecting cars and recognizing their brand and model in images. Includes data augmentation, validation with no-car samples, and full training pipeline.

---

## Project Overview

The goal of this project is to build a Convolutional Neural Network (CNN) using TensorFlow/Keras to detect cars and recognize their brand/model in images. The network is trained using a car dataset (e.g., from Kaggle), with image augmentation and validation that includes samples without cars.

**Supported Car Classes:**
- Aston Martin V8 Vantage Convertible 2012
- Audi TT RS Coupe 2012
- BMW X6 SUV 2012
- Ferrari 458 Italia Convertible 2012
- Lamborghini Aventador Coupe 2012
- Nissan Juke Hatchback 2012
- Smart Fortwo Convertible 2012
- Toyota Camry Sedan 2012
- No Car (negative samples)

### Dataset Sources

The dataset used in this project was obtained from:

- **Car Images:** [Stanford Car Dataset by Classes Folder](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder) on Kaggle
  - For faster training, I manually removed some car brands and kept only the ones I liked the most (8 classes)
  
- **No-Car Images:** [Natural Images Dataset](https://www.kaggle.com/datasets/prasunroy/natural-images/data) on Kaggle
  - Used to create the `no_car` class for negative samples

---

## About Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are a class of deep learning models specifically designed for processing structured grid data like images. They automatically learn spatial hierarchies of features through backpropagation, making them highly effective for image recognition tasks such as car brand classification.

![CNN Architecture](images/cnn_architecture.png)

### Key CNN Concepts

- **Convolution Layers** — Apply learnable filters to input images to detect features like edges, textures, and patterns
- **Pooling Layers** — Reduce spatial dimensions while retaining important features (e.g., MaxPooling)
- **Activation Functions (ReLU)** — Introduce non-linearity, allowing the network to learn complex patterns
- **Fully Connected (Dense) Layers** — Combine features from previous layers to make final predictions
- **Dropout Regularization** — Randomly deactivate neurons during training to prevent overfitting
- **Transfer Learning (MobileNetV2)** — Leverage pre-trained weights from ImageNet to improve accuracy with limited data

---

## Project Implementation

This project uses **MobileNetV2** as a pre-trained base model for feature extraction. MobileNetV2 is a lightweight, efficient architecture optimized for mobile and embedded vision applications.

### Data Augmentation

The training pipeline uses `ImageDataGenerator` with the following augmentations:

```python
train_datagener = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
```

### Training Phases

Training is conducted in two phases:
1. **Phase 1:** Base model layers are frozen; only the custom classification head is trained
2. **Phase 2:** Fine-tuning with unfrozen layers for improved accuracy

### Callbacks

- **ModelCheckpoint** — Saves the best model based on validation accuracy
- **EarlyStopping** — Stops training when validation loss stops improving (patience=8)
- **ReduceLROnPlateau** — Reduces learning rate when validation loss plateaus

---

## Code Explanation with Snippets

### Model Architecture (`cnn()` function)

```python
def cnn(input_shape, num_classes):
    # Use MobileNetV2 as base model (pre-trained on ImageNet)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model layers initially
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
```

**Explanation:**
- Loads MobileNetV2 with pre-trained ImageNet weights (excluding the top classification layer)
- Freezes base model layers to preserve learned features
- Adds a custom head: GlobalAveragePooling → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Softmax output

### Training Process (`train_model()`)

```python
def train_model(model, train_gener, validate_gener, epochs, model_path):
    # Phase 1: Train with frozen base model
    print("\n--- Phase 1: Training with frozen base model ---")
    history1 = model.fit(
        train_gener,
        validation_data=validate_gener,
        epochs=epochs // 2,
        callbacks=[checkpoint, early, reduce_lr]
    )
    
    # Phase 2: Fine-tune by unfreezing some layers
    print("\n--- Phase 2: Fine-tuning with unfrozen layers ---")
    model = unfreeze_model(model, num_layers_to_unfreeze=50)
    model.compile(
        optimizer=Adam(learning_rate=lr / 10),  # Lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_gener,
        validation_data=validate_gener,
        epochs=epochs,
        callbacks=[checkpoint2, early2, reduce_lr2]
    )
    
    return history
```

**Explanation:**
- **Phase 1:** Trains only the custom classification head with frozen MobileNetV2 layers
- **Phase 2:** Unfreezes the last 50 layers and fine-tunes with a reduced learning rate (lr/10) for stable training

### Prediction (`prediction()` and `print_predictions()`)

```python
def prediction(model, img_path, img_size=(224, 224), class_indices=None, top_k=3):
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    
    # Get top-k predictions
    top_indices = preds.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        cls = inv_map.get(idx, str(idx))
        prob = float(preds[idx])
        results.append((cls, prob))
    
    return best_cls, best_prob, results


def print_predictions(results):
    print("\n" + "="*50)
    print("PREDICTION RESULTS:")
    print("="*50)
    for i, (cls, prob) in enumerate(results, 1):
        bar = "█" * int(prob * 30)
        print(f"{i}. {cls}")
        print(f"   Confidence: {prob*100:.2f}% {bar}")
    print("="*50)
```

**Explanation:**
- Loads and preprocesses the image (resize, normalize)
- Returns the top-k predictions with confidence scores
- Displays results with visual confidence bars

---

## Training & Validation

### Two-Stage Training

| Phase | Description | Learning Rate |
|-------|-------------|---------------|
| Phase 1 | Base model frozen, train classification head only | 0.0001 |
| Phase 2 | Unfreeze last 50 layers, fine-tune entire model | 0.00001 |

### Validation with No-Car Samples

The validation set includes a `no_car` folder with images that don't contain cars. This helps the model learn to distinguish between images with and without cars, reducing false positives.

---

## Evaluation and Prediction

After training, the model can be evaluated and used for predictions:

```python
# Evaluate on validation set
results = model.evaluate(validate_gener)
print("Validation results:", results)

# Make predictions on new images
model = load_model(model_path)
cls, prob, all_results = prediction(model, "path/to/image.jpg", 
                                     img_size=img_size, 
                                     class_indices=train_gen.class_indices, 
                                     top_k=3)
print_predictions(all_results)
```

**Sample Output:**
```
==================================================
PREDICTION RESULTS:
==================================================
1. Ferrari 458 Italia Convertible 2012
   Confidence: 94.32% ████████████████████████████
2. Lamborghini Aventador Coupe 2012
   Confidence: 3.21% █
3. Aston Martin V8 Vantage Convertible 2012
   Confidence: 1.15% 
==================================================
```

---

## Results Visualization

The `plot_history()` function visualizes training progress:

```python
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title("Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title("Loss")
    plt.show()
```

### Training Results

Below are the actual training results from the model:

![Training Results](images/training_results.png)

**Training Statistics:**

| Metric | Final Value |
|--------|-------------|
| Training Accuracy | ~93% |
| Validation Accuracy | ~99% |
| Training Loss | ~0.25 |
| Validation Loss | ~0.18 |
| Total Epochs | ~75 (Phase 1: 25 + Phase 2: 50) |

**Observations:**
- The model shows excellent convergence with validation accuracy reaching ~99%
- Phase 1 (epochs 1-25): Training with frozen base model
- Phase 2 (epochs 26-75): Fine-tuning with unfrozen layers, showing significant improvement
- The gap between training and validation accuracy indicates good generalization
- Loss curves show steady improvement throughout both training phases

---

## Setup & Usage Instructions

### Prerequisites

Install the required dependencies:

```bash
pip install tensorflow keras numpy matplotlib pillow
```

### Project Structure

```
Car-Brand-Recognition-CNN/
├── car_detector.py          # Main training and prediction script
├── trained_model.h5         # Saved model weights
├── train/                   # Training images
│   ├── Aston Martin V8 Vantage Convertible 2012/
│   ├── Audi TT RS Coupe 2012/
│   ├── BMW X6 SUV 2012/
│   ├── Ferrari 458 Italia Convertible 2012/
│   ├── Lamborghini Aventador Coupe 2012/
│   ├── Nissan Juke Hatchback 2012/
│   ├── no_car/
│   ├── smart fortwo Convertible 2012/
│   └── Toyota Camry Sedan 2012/
└── validate/                # Validation images (same structure)
```

### How to Run

1. **Prepare your dataset** — Place training images in `train/` and validation images in `validate/`, organized by class folders

2. **Train the model:**
   ```bash
   python car_detector.py
   ```
   Uncomment the training section in the `__main__` block before running.

3. **Make predictions:**
   ```python
   model = load_model("trained_model.h5")
   cls, prob, all_results = prediction(model, "path/to/car_image.jpg", 
                                        img_size=(224, 224), 
                                        class_indices=train_gen.class_indices)
   print_predictions(all_results)
   ```

### Configuration

Adjust these parameters in `car_detector.py` as needed:

```python
train_dir = "train"           # Training data folder
validate_dir = "validate"     # Validation data folder
img_size = (224, 224)         # Input image size
batch_size = 32               # Batch size for training
epochs = 50                   # Maximum training epochs
model_path = "trained_model.h5"  # Output model file
lr = 0.0001                   # Initial learning rate
```

---

## License and Credits

- **TensorFlow/Keras** — Deep learning framework used for model development
- **MobileNetV2** — Pre-trained model architecture from Google for transfer learning
- **ImageNet** — Dataset used for pre-training MobileNetV2 weights

This project was developed as a university assignment to demonstrate CNN-based image classification using transfer learning techniques.

---

## Author

Built with TensorFlow/Keras for educational purposes.
