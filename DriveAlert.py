import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
import cv2
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk
import pygame
import threading
import time
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape
import seaborn as sns

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Cargar modelos existentes ---
if os.path.exists("models/eye_model_trained.h5"):
    eye_classifier = load_model("models/eye_model_trained.h5")
    print("Modelo de ojos cargado desde archivo.")
else:
    print("Entrenando modelo de ojos...")

    eye_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    eye_model.trainable = False

    eye_classifier = Sequential([
        eye_model,
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),

        GlobalAveragePooling2D(),

        Dense(512, activation='relu'),
        Dropout(0.6),
        BatchNormalization(),

        Dense(256, activation='relu'),
        Dropout(0.6),
        BatchNormalization(),

        Dense(128, activation='relu'),
        Dropout(0.6),
        BatchNormalization(),

        Dense(64, activation='relu'),
        Dropout(0.4),

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

    eye_classifier.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    eye_classifier.fit(
        eye_train_generator,
        validation_data=eye_val_generator,
        epochs=1
    )

    os.makedirs("models", exist_ok=True)
    eye_classifier.save("models/eye_model_trained.h5")
    print("Modelo de ojos entrenado y guardado.")

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

if os.path.exists("models/yawn_model_trained.h5"):
    yawn_model = load_model("models/yawn_model_trained.h5")
    print("Modelo de bostezo cargado desde archivo.")
else:
    print("Entrenando modelo de bostezo...")

    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    yawn_model = Sequential([
        base_model,
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),

        GlobalAveragePooling2D(),

        Dense(512, activation='relu'),
        Dropout(0.2),
        BatchNormalization(),

        Dense(256, activation='relu'),
        Dropout(0.1),
        BatchNormalization(),

        Dense(128, activation='relu'),
        Dropout(0.1),
        BatchNormalization(),

        Dense(64, activation='relu'),
        Dropout(0.1),

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
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        'yawn_model',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        'yawn_model',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    yawn_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    yawn_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10
    )

    os.makedirs("models", exist_ok=True)
    yawn_model.save("models/yawn_model_trained.h5")
    print("Modelo de bostezo entrenado y guardado.")

    val_generator.reset()
    yawn_pred_probs = yawn_model.predict(val_generator)
    yawn_pred = (yawn_pred_probs > 0.5).astype(int).flatten()
    yawn_true = val_generator.classes

    print("Reporte de Clasificación - Modelo de Bostezo:")
    print(classification_report(yawn_true, yawn_pred, target_names=list(val_generator.class_indices.keys())))

    cm_yawn = confusion_matrix(yawn_true, yawn_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_yawn, annot=True, fmt='d', cmap='Purples',
                xticklabels=val_generator.class_indices.keys(),
                yticklabels=val_generator.class_indices.keys())
    plt.title("Matriz de Confusión - Modelo de Bostezo")
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.show()

# --- Funciones de predicción ---
def predict_yawn(image):
    img = cv2.resize(image, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return yawn_model.predict(img)[0][0]

def predict_eye(image):
    img = cv2.resize(image, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return eye_classifier.predict(img)[0][0]

# --- NUEVA FUNCIÓN: Detección combinada de ojos ---
def detect_eyes_combined(frame, eye_threshold=0.2):
    """
    Combina detección de OpenCV con modelo entrenado usando lógica OR
    
    OJOS ABIERTOS: Si modelo predice abiertos OR OpenCV detecta ojos
    OJOS CERRADOS: Si modelo predice cerrados AND OpenCV no detecta ojos
    
    Returns:
        - eyes_open (bool): True si ojos están abiertos
        - model_prediction (float): Predicción del modelo
        - opencv_detected (bool): True si OpenCV detectó ojos
        - eyes_rectangles (list): Coordenadas de ojos detectados por OpenCV
    """
    
    # Predicción del modelo
    img_resized = cv2.resize(frame, (224, 224))
    model_pred = predict_eye(img_resized)
    model_says_open = model_pred > eye_threshold
    
    # Detección con OpenCV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    opencv_detected = len(eyes) > 0
    
    # LÓGICA COMBINADA:
    # Ojos ABIERTOS: Si modelo dice abierto OR OpenCV detecta ojos
    eyes_open = model_says_open or opencv_detected
    
    return eyes_open, model_pred, opencv_detected, eyes

# --- Interfaz Tkinter sin calibración ---
class SleepDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Sueño")

        self.eyes_closed_start_time = None
        self.eyes_closed_duration_threshold = 5

        # Umbrales predefinidos (sin calibración)
        self.thresholds = {
            "eye_threshold": 0.8,    # Umbral para ojos (0.5 es un valor estándar)
            "yawn_threshold": 0.4    # Umbral para bostezo (0.5 es un valor estándar)
        }
        
        print(f"Usando detección combinada:")
        print(f"- Ojos ABIERTOS: Modelo > {self.thresholds['eye_threshold']} OR OpenCV detecta ojos")
        print(f"- Ojos CERRADOS: Modelo < {self.thresholds['eye_threshold']} AND OpenCV no detecta ojos")
        print(f"- Bostezo: Modelo > {self.thresholds['yawn_threshold']}")

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.status_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.status_label.pack(pady=10)

        # Etiqueta para mostrar detalles de la detección
        self.details_label = tk.Label(root, text="", font=("Helvetica", 10), fg="gray")
        self.details_label.pack(pady=5)

        self.close_button = tk.Button(root, text="Cerrar", command=self.close_app)
        self.close_button.pack()

        self.cap = cv2.VideoCapture(0)

        pygame.mixer.init()
        pygame.mixer.music.load("alerta.mp3")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.update_video()

    def play_alert_sound(self):
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            # Predicción de bostezo (sin cambios)
            img_resized = cv2.resize(frame, (224, 224))
            yawn_pred = predict_yawn(img_resized)
            yawn_detected = yawn_pred > self.thresholds["yawn_threshold"]
            
            # NUEVA DETECCIÓN COMBINADA DE OJOS
            eyes_open, model_pred, opencv_detected, eyes_rectangles = detect_eyes_combined(
                frame, self.thresholds["eye_threshold"]
            )

            # Textos para mostrar
            yawn_text = "Bostezo" if yawn_detected else "No bostezo"
            eye_text = "Ojos abiertos" if eyes_open else "Ojos cerrados"
            
            # Detalles de la detección combinada
            model_status = f"Modelo: {model_pred:.2f} ({'Abierto' if model_pred > self.thresholds['eye_threshold'] else 'Cerrado'})"
            opencv_status = f"OpenCV: {'Detectado' if opencv_detected else 'No detectado'}"
            combined_status = f"Resultado: {eye_text}"
            details_text = f"{model_status} | {opencv_status} | {combined_status}"

            # Alertas
            if yawn_detected:
                threading.Thread(target=self.play_alert_sound, daemon=True).start()

            # Control de tiempo de ojos cerrados
            if not eyes_open:  # Ojos cerrados según lógica combinada
                if self.eyes_closed_start_time is None:
                    self.eyes_closed_start_time = time.time()
                elif time.time() - self.eyes_closed_start_time >= self.eyes_closed_duration_threshold:
                    threading.Thread(target=self.play_alert_sound, daemon=True).start()
            else:
                self.eyes_closed_start_time = None

            # Dibujar rectángulos de rostros
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
                cv2.putText(frame, "Rostro", (fx, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Mostrar texto en el frame
            cv2.putText(frame, f"{yawn_text} ({yawn_pred:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"{eye_text} (M:{model_pred:.2f}, CV:{len(eyes_rectangles)})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Actualizar etiquetas de la interfaz
            self.status_label.config(text=f"{yawn_text} | {eye_text}")
            self.details_label.config(text=details_text)

        else:
            # No hay rostros detectados
            yawn_text = ""
            eye_text = ""
            self.eyes_closed_start_time = None
            
            cv2.putText(frame, "Rostro no detectado", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            self.status_label.config(text="Rostro no detectado")
            self.details_label.config(text="")

        # Mostrar frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_video)

    def close_app(self):
        self.cap.release()
        self.root.destroy()

# --- Ejecutar interfaz ---
if __name__ == "__main__":
    root = tk.Tk()
    app = SleepDetectorApp(root)
    root.mainloop()