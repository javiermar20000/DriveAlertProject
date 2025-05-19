import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
import shutil

def load_or_train_yawn_model():
    if os.path.exists("models/yawn_model_trained.h5"):
        yawn_model = load_model("models/yawn_model_trained.h5")
        print("Modelo de bostezo cargado desde archivo.")
    else:
        print("Entrenando modelo de bostezo...")

        yawn_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        yawn_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Usar tus rutas de datos
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            'yawn_model/train',
            target_size=(64, 64),
            batch_size=16,
            class_mode='binary'
        )
        val_generator = test_datagen.flow_from_directory(
            'yawn_model/test',
            target_size=(64, 64),
            batch_size=16,
            class_mode='binary',
            shuffle=False
        )

        yawn_model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=4
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
    img = cv2.resize(image, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return yawn_model.predict(img)[0][0]

def balancear_clases(directorio):
    clases = [d for d in os.listdir(directorio) if os.path.isdir(os.path.join(directorio, d))]
    conteos = {}
    for clase in clases:
        ruta = os.path.join(directorio, clase)
        conteos[clase] = [os.path.join(ruta, f) for f in os.listdir(ruta) if os.path.isfile(os.path.join(ruta, f))]
    min_cantidad = min(len(imgs) for imgs in conteos.values())
    for clase, imgs in conteos.items():
        if len(imgs) > min_cantidad:
            eliminar = random.sample(imgs, len(imgs) - min_cantidad)
            for img in eliminar:
                os.remove(img)
            print(f"Se eliminaron {len(eliminar)} imágenes de la clase '{clase}' para balancear.")

print("Balanceando train:")
balancear_clases('yawn_model/train')
print("Balanceando test:")
balancear_clases('yawn_model/test')
