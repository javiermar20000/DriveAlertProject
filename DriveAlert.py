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

# --- Cargar modelos existentes ---
if os.path.exists("models/eye_model_trained.h5"):
    eye_classifier = load_model("models/eye_model_trained.h5")
    print("Modelo de ojos cargado desde archivo.")
else:
    print("Entrenando modelo de ojos...")
    eye_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
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
        subset='validation'
    )

    eye_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    eye_classifier.fit(eye_train_generator, validation_data=eye_val_generator, epochs=30)

    os.makedirs("models", exist_ok=True)
    eye_classifier.save("models/eye_model_trained.h5")
    print("Modelo de ojos entrenado y guardado.")

if os.path.exists("models/yawn_model_trained.h5"):
    yawn_model = load_model("models/yawn_model_trained.h5")
    print("Modelo de bostezo cargado desde archivo.")
else:
    print("Entrenando modelo de bostezo...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
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
    fill_mode='nearest',
    validation_split=0.2
)

    train_generator = train_datagen.flow_from_directory(
        'yawn_model',
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        'yawn_model',
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        subset='validation'
    )

    yawn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    yawn_model.fit(train_generator, validation_data=val_generator, epochs=30)

    os.makedirs("models", exist_ok=True)
    yawn_model.save("models/yawn_model_trained.h5")
    print("Modelo de bostezo entrenado y guardado.")

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

# --- Calibración inicial del usuario ---
def calibrate_user(cap):
    def capture_state(prompt, seconds=3, samples=15):
        print(f"\n[Instrucción] {prompt}")
        print("Capturando imágenes en...")
        for i in range(seconds, 0, -1):
            print(f"{i}...")
            time.sleep(1)

        preds = []
        for _ in range(samples):
            ret, frame = cap.read()
            if ret:
                if "ojos" in prompt.lower():
                    preds.append(predict_eye(frame))
                elif "bostezo" in prompt.lower():
                    preds.append(predict_yawn(frame))
            time.sleep(0.3)

            preds = sorted(preds)
            trimmed = preds[2:-2]  # eliminar los 2 valores más bajos y más altos
        return np.mean(trimmed)

    print("\n=== Calibración de usuario ===")
    open_eye_vals = [capture_state("MANTÉN LOS OJOS ABIERTOS")]
    closed_eye_vals = [capture_state("CIERRA LOS OJOS COMPLETAMENTE")]
    yawn_vals = [capture_state("BOSTEZA (o simula un bostezo fuerte)")]
    no_yawn_vals = [capture_state("RELÁJATE SIN BOSTEZAR (posición neutral)")]

    eye_threshold = (np.percentile(closed_eye_vals, 25) + np.percentile(open_eye_vals, 75)) / 2
    yawn_threshold = (np.percentile(no_yawn_vals, 25) + np.percentile(yawn_vals, 75)) / 2

    print("\n--- Calibración completada ---")
    return {
        "eye_threshold": eye_threshold,
        "yawn_threshold": yawn_threshold
    }

# --- Interfaz Tkinter con calibración integrada ---
class SleepDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Sueño")

        self.eyes_closed_start_time = None
        self.eyes_closed_duration_threshold = 5  # segundos
        
        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.status_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.status_label.pack(pady=10)

        self.close_button = tk.Button(root, text="Cerrar", command=self.close_app)
        self.close_button.pack()

        self.cap = cv2.VideoCapture(0)

        pygame.mixer.init()
        pygame.mixer.music.load("alerta.mp3")

        # Calibración del usuario
        self.thresholds = calibrate_user(self.cap)

        self.update_video()

    def play_alert_sound(self):
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        img_resized = cv2.resize(frame, (224, 224))
        yawn_pred = predict_yawn(img_resized)
        eye_pred = predict_eye(img_resized)

        yawn_text = "Bostezo" if yawn_pred > self.thresholds["yawn_threshold"] else "No bostezo"
        eye_text = "Ojos abiertos" if eye_pred > self.thresholds["eye_threshold"] else "Ojos cerrados"

        # Bostezo: activar alarma de inmediato
        if yawn_pred > self.thresholds["yawn_threshold"]:
            threading.Thread(target=self.play_alert_sound, daemon=True).start()

        # Ojos cerrados: activar solo si han pasado 5 segundos seguidos
        if eye_pred < self.thresholds["eye_threshold"]:
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = time.time()
            elif time.time() - self.eyes_closed_start_time >= self.eyes_closed_duration_threshold:
                threading.Thread(target=self.play_alert_sound, daemon=True).start()
        else:
            self.eyes_closed_start_time = None  # Reiniciar si abre los ojos

        # Mostrar texto sobre el frame
        cv2.putText(frame, f"{yawn_text} ({yawn_pred:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"{eye_text} ({eye_pred:.2f})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        self.status_label.config(text=f"{yawn_text} | {eye_text}")

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