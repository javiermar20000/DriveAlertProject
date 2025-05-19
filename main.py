import cv2
import tkinter as tk
from PIL import Image, ImageTk
import pygame
import threading
import time

from eye_detector import load_or_train_eye_model, predict_eye
from yawn_detector import load_or_train_yawn_model, predict_yawn
from calibration import calibrate_user

class SleepDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Sueño")
        self.eyes_closed_start_time = None
        self.eyes_closed_duration_threshold = 5
        self.video_label = tk.Label(root)
        self.video_label.pack()
        self.status_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.status_label.pack(pady=10)
        self.close_button = tk.Button(root, text="Cerrar", command=self.close_app)
        self.close_button.pack()
        self.cap = cv2.VideoCapture(0)
        pygame.mixer.init()
        pygame.mixer.music.load("alerta.mp3")

        # Cargar modelos
        self.eye_classifier = load_or_train_eye_model()
        self.yawn_model = load_or_train_yawn_model()

        # Calibración
        self.thresholds = calibrate_user(
            self.cap,
            lambda img: predict_eye(img, self.eye_classifier),
            lambda img: predict_yawn(img, self.yawn_model)
        )

        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.update_video()

    def play_alert_sound(self):
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        img_resized = cv2.resize(frame, (224, 224))
        yawn_pred = predict_yawn(img_resized, self.yawn_model)
        eye_pred = predict_eye(img_resized, self.eye_classifier)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            yawn_text = "Bostezo" if yawn_pred > self.thresholds["yawn_threshold"] else "No bostezo"
            eye_text = "Ojos abiertos" if eye_pred > self.thresholds["eye_threshold"] else "Ojos cerrados"
            if yawn_pred > self.thresholds["yawn_threshold"]:
                threading.Thread(target=self.play_alert_sound, daemon=True).start()
            if eye_pred < self.thresholds["eye_threshold"]:
                if self.eyes_closed_start_time is None:
                    self.eyes_closed_start_time = time.time()
                elif time.time() - self.eyes_closed_start_time >= self.eyes_closed_duration_threshold:
                    threading.Thread(target=self.play_alert_sound, daemon=True).start()
            else:
                self.eyes_closed_start_time = None
        else:
            yawn_text = ""
            eye_text = ""
            self.eyes_closed_start_time = None

        # Detección de ojos
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(frame, "Ojos detectados", (ex, ey + eh + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Detección de rostros
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
            cv2.putText(frame, "Rostro detectado", (fx, fy + fh + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(faces) == 0:
            cv2.putText(frame, "Rostro no detectado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"{yawn_text} ({yawn_pred:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"{eye_text} ({eye_pred:.2f})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        if len(faces) == 0:
            self.status_label.config(text="Rostro no detectado")
        else:
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

if __name__ == "__main__":
    root = tk.Tk()
    app = SleepDetectorApp(root)
    root.mainloop()