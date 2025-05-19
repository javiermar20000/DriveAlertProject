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

def load_or_train_yawn_model():
    if os.path.exists("models/yawn_model_trained.h5"):
        yawn_model = load_model("models/yawn_model_trained.h5")
        print("Modelo de bostezo cargado desde archivo.")
    else:
        print("Entrenando modelo de bostezo...")
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        yawn_model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(1, activation='sigmoid')
        ])
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        train_generator = train_datagen.flow_from_directory(
            'yawn_model/train',
            target_size=(224, 224),
            batch_size=16,
            class_mode='binary'
        )
        val_generator = test_datagen.flow_from_directory(
            'yawn_model/test',
            target_size=(224, 224),
            batch_size=16,
            class_mode='binary'
        )
        yawn_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        yawn_model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=30
        )
        os.makedirs("models", exist_ok=True)
        yawn_model.save("models/yawn_model_trained.h5")
        print("Modelo de bostezo entrenado y guardado.")

        # Evaluación
        val_generator.reset()
        yawn_pred_probs = yawn_model.predict(val_generator)
        yawn_pred = (yawn_pred_probs > 0.5).astype(int).flatten()
        yawn_true = val_generator.classes
        print("Reporte de Clasificación - Modelo de Bostezo:")
        print(classification_report(yawn_true, yawn_pred, target_names=list(val_generator.class_indices.keys())))
        cm_yawn = confusion_matrix(yawn_true, yawn_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm_yawn, annot=True, fmt='d', cmap='Purples', xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
        plt.title("Matriz de Confusión - Modelo de Bostezo")
        plt.xlabel("Predicción")
        plt.ylabel("Verdadero")
        plt.show()
    return yawn_model

def predict_yawn(image, yawn_model):
    img = cv2.resize(image, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return yawn_model.predict(img)[0][0]