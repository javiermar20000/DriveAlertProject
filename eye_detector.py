import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
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

        # Arquitectura más profunda
        eye_classifier = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2,2)),

            Conv2D(64, (3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2,2)),

            Conv2D(128, (3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2,2)),

            Conv2D(256, (3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2,2)),

            Conv2D(512, (3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2,2)),

            Flatten(),

            Dense(512, activation='relu'),
            Dropout(0.5),

            Dense(256, activation='relu'),
            Dropout(0.3),

            Dense(128, activation='relu'),

            Dense(1, activation='sigmoid')
        ])

        # Preprocesamiento y aumento de datos
        eye_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
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
            batch_size=32,
            class_mode='binary',
            subset='training'
        )

        eye_val_generator = eye_datagen.flow_from_directory(
            'eyes_model',
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )

        # Compilación
        eye_classifier.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Entrenamiento
        eye_classifier.fit(
            eye_train_generator,
            validation_data=eye_val_generator,
            epochs=10  # Aumenta las épocas si tienes suficientes datos
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=eye_val_generator.class_indices.keys(),
                    yticklabels=eye_val_generator.class_indices.keys())
        plt.title("Matriz de Confusión - Modelo de Ojos")
        plt.xlabel("Predicción")
        plt.ylabel("Verdadero")
        plt.show()

    return eye_classifier

def predict_eye(image, eye_classifier):
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return eye_classifier.predict(img)[0][0]