import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_or_train_eye_model():
    if os.path.exists("models/eye_model_trained.h5"):
        eye_classifier = load_model("models/eye_model_trained.h5")
        print("Modelo de ojos cargado desde archivo.")
    else:
        print("Entrenando modelo de ojos...")
        eye_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        eye_model.trainable = False
        eye_classifier = Sequential([
            eye_model,
            GlobalAveragePooling2D(),
            Dense(1, activation='sigmoid')
        ])
        eye_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        eye_train_generator = eye_datagen.flow_from_directory(
            'eyes_model',
            target_size=(224, 224),
            batch_size=16,
            class_mode='binary',
            subset='training'
        )
        eye_val_generator = eye_datagen.flow_from_directory(
            'eyes_model',
            target_size=(224, 224),
            batch_size=16,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        eye_classifier.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        eye_classifier.fit(
            eye_train_generator,
            validation_data=eye_val_generator,
            epochs=2
        )
        os.makedirs("models", exist_ok=True)
        eye_classifier.save("models/eye_model_trained.h5")
        print("Modelo de ojos entrenado y guardado.")

        # Evaluación
        eye_val_generator.reset()
        y_pred_probs = eye_classifier.predict(eye_val_generator)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = eye_val_generator.classes
        print("Reporte de Clasificación - Modelo de Ojos:")
        print(classification_report(y_true, y_pred, target_names=list(eye_val_generator.class_indices.keys())))
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=eye_val_generator.class_indices.keys(), yticklabels=eye_val_generator.class_indices.keys())
        plt.title("Matriz de Confusión - Modelo de Ojos")
        plt.xlabel("Predicción")
        plt.ylabel("Verdadero")
        plt.show()
    return eye_classifier

def predict_eye(image, eye_classifier):
    img = cv2.resize(image, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return eye_classifier.predict(img)[0][0]
