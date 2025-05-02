import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

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
    raise FileNotFoundError("No se encontró el modelo de ojos entrenado.")

if os.path.exists("models/yawn_model_trained.h5"):
    yawn_model = load_model("models/yawn_model_trained.h5")
    print("Modelo de bostezo cargado desde archivo.")
else:
    raise FileNotFoundError("No se encontró el modelo de bostezo entrenado.")

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
    def capture_state(prompt, seconds=3, samples=5):
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
        return np.mean(preds)

    print("\n=== Calibración de usuario ===")
    open_eye_thresh = capture_state("MANTÉN LOS OJOS ABIERTOS")
    closed_eye_thresh = capture_state("CIERRA LOS OJOS COMPLETAMENTE")
    yawn_thresh = capture_state("BOSTEZA (o simula un bostezo fuerte)")
    no_yawn_thresh = capture_state("RELÁJATE SIN BOSTEZAR (posición neutral)")

    print("\n--- Calibración completada ---")
    return {
        "eye_threshold": (open_eye_thresh + closed_eye_thresh) / 2,
        "yawn_threshold": (yawn_thresh + no_yawn_thresh) / 2
    }

# --- Interfaz Tkinter con calibración integrada ---
class SleepDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Sueño")
        
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

        if yawn_pred > self.thresholds["yawn_threshold"] or eye_pred < self.thresholds["eye_threshold"]:
            threading.Thread(target=self.play_alert_sound, daemon=True).start()

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