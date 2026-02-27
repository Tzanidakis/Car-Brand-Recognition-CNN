import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2


#------configure metrics--------
train_dir = "train"  # folder name where the data is stored
validate_dir = "validate"  # folder name where the data is stored
img_size = (224, 224)  # larger images for better car recognition
batch_size = 32  # batch size
epochs = 50  # enough epochs with early stopping
model_path = "trained_model.h5"  # trained model name
lr = 0.0001  # lower learning rate for transfer learning
#--------------------------------


def generators(train_dir, validate_dir, img_size, batch_size):
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
    train_gener = train_datagener.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    validate_datagener = ImageDataGenerator(rescale=1./255)
    validate_gener = validate_datagener.flow_from_directory(
        validate_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gener, validate_gener


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


def unfreeze_model(model, num_layers_to_unfreeze=30):
    """Unfreeze some layers of the base model for fine-tuning"""
    # Find the base model (MobileNetV2)
    for layer in model.layers:
        layer.trainable = True
    
    # Keep first layers frozen, unfreeze last num_layers_to_unfreeze
    for layer in model.layers[:-num_layers_to_unfreeze]:
        if not isinstance(layer, (Dense, Dropout)):
            layer.trainable = False
    return model


def compile_model(model):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(model, train_gener, validate_gener, epochs, model_path):
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

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
    
    # Reset callbacks for phase 2
    checkpoint2 = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early2 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    
    history2 = model.fit(
        train_gener,
        validation_data=validate_gener,
        epochs=epochs,
        callbacks=[checkpoint2, early2, reduce_lr2]
    )
    
    # Combine histories
    history = type('obj', (object,), {'history': {}})()
    for key in history1.history:
        history.history[key] = history1.history[key] + history2.history[key]
    
    return history


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


def evaluate_model(model, validate_gener):
    results = model.evaluate(validate_gener)
    print("Validation results:", results)
    return results


def prediction(model, img_path, img_size=(224, 224), class_indices=None, top_k=3):
    """
    Predict the class of an image and return top-k predictions with probabilities.
    """
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]  # Get predictions for single image
    
    # Create inverse mapping
    inv_map = {v: k for k, v in (class_indices or {}).items()}
    
    # Get top-k predictions
    top_indices = preds.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        cls = inv_map.get(idx, str(idx))
        prob = float(preds[idx])
        results.append((cls, prob))
    
    # Return best prediction and all top-k results
    best_cls, best_prob = results[0]
    return best_cls, best_prob, results


def print_predictions(results):
    """Print top predictions in a nice format"""
    print("\n" + "="*50)
    print("PREDICTION RESULTS:")
    print("="*50)
    for i, (cls, prob) in enumerate(results, 1):
        bar = "█" * int(prob * 30)
        print(f"{i}. {cls}")
        print(f"   Confidence: {prob*100:.2f}% {bar}")
    print("="*50)


if __name__ == "__main__":
    train_gen, val_gen = generators(train_dir, validate_dir, img_size, batch_size)
    num_classes = train_gen.num_classes
    print(f"Found {num_classes} classes.")
    print("Classes:", list(train_gen.class_indices.keys()))

    model = cnn(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    model = compile_model(model)
    model.summary()
    '''
    # Train the model (uncomment load_model section after training is complete)
    history = train_model(model, train_gen, val_gen, model_path=model_path, epochs=epochs)
    plot_history(history)
    evaluate_model(model, val_gen)
    '''
    # Uncomment after training to make predictions
    '''
    model = load_model(model_path)
    cls, prob, all_results = prediction(model, "image_for_detection", img_size=img_size, class_indices=train_gen.class_indices, top_k=3)
    print_predictions(all_results)
    print(f"\nBest prediction: {cls} ({prob*100:.2f}%)")
    '''
    