import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import time
from collections import deque
import math
import pygame
import threading
import os

class EnhancedDrowsinessDetector:
    def __init__(self):
        # Inicializar MediaPipe Face Mesh (mismo que el código original)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Inicializar sistema de audio
        self.init_audio_system()
        
        # USAR EXACTAMENTE LOS MISMOS LANDMARKS DEL CÓDIGO ORIGINAL
        # Índices de landmarks para MediaPipe Face Mesh
        # Ojos (usando landmarks específicos de MediaPipe)
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Puntos específicos para cálculo EAR (EXACTOS DEL ORIGINAL)
        self.LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]  # Equivalentes a dlib
        self.RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
        
        # Boca - Puntos más completos para mejor visualización (EXACTOS DEL ORIGINAL)
        # Contorno exterior completo de la boca
        self.MOUTH_OUTER_LANDMARKS = [
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            291, 303, 267, 269, 270, 271, 272, 302, 268, 271, 272
        ]
        
        # Contorno interior de la boca (labios)
        self.MOUTH_INNER_LANDMARKS = [
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
            310, 311, 312, 13, 82, 81, 80, 78
        ]
        
        # Puntos específicos para detección (esquinas y centro)
        self.MOUTH_KEY_POINTS = [
            61, 291,    # Esquinas izquierda y derecha
            13, 14,     # Centro superior e inferior
            78, 308,    # Labio superior centro
            87, 317,    # Puntos laterales superiores
            84, 314,    # Puntos externos superiores
            17, 18,     # Centro muy superior e inferior
            200, 199    # Puntos adicionales centro
        ]
        
        # Todos los puntos de boca combinados para visualización (EXACTOS DEL ORIGINAL)
        self.ALL_MOUTH_LANDMARKS = list(set(
            self.MOUTH_OUTER_LANDMARKS + 
            self.MOUTH_INNER_LANDMARKS + 
            self.MOUTH_KEY_POINTS + 
            [0, 11, 12, 15, 16, 17, 18, 200, 199, 175, 176, 177, 
             180, 181, 184, 185, 191, 192, 193, 194, 204, 205, 206, 
             207, 210, 211, 212, 213, 216, 217, 218, 219, 220, 305, 
             306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 
             317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 
             375, 376, 377, 378, 379, 380, 381, 382, 402, 403, 404, 
             405, 406, 407, 408, 409, 410, 411, 412, 415, 416, 417, 
             418, 419, 420, 421, 422, 424, 425, 426, 427, 428, 429, 
             430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440]
        ))
        
        self.MOUTH_EAR = [13, 14, 12, 15, 268, 269, 271, 272]  # Puntos para MAR
        
        # Umbrales (USAR LOS MISMOS DEL CÓDIGO ORIGINAL)
        self.EYE_AR_THRESH = 0.25
        self.MOUTH_AR_THRESH = 0.5  # Usar el umbral original más bajo
        self.EYE_AR_CONSEC_FRAMES = 3
        
        # Umbrales para alertas (más conservadores para reducir falsos positivos)
        self.ALERT_LEVEL_1 = 50  # Alerta suave
        self.ALERT_LEVEL_2 = 70  # Alerta media
        self.ALERT_LEVEL_3 = 85  # Alerta fuerte
        
        # Contadores (EXACTOS DEL ORIGINAL)
        self.eye_counter = 0
        self.yawn_counter = 0
        self.total_blinks = 0
        self.total_yawns = 0
        self.yawn_detected = False
        
        # Sistema de alertas
        self.current_alert_level = 0
        self.alert_start_time = None
        self.last_alert_time = 0
        self.alert_cooldown = 8  # Más tiempo entre alertas para evitar molestias
        self.is_playing_alert = False
        self.alert_thread = None
        
        # Historial para análisis temporal (más conservador)
        self.ear_history = deque(maxlen=150)  # 5 segundos a 30fps
        self.mouth_history = deque(maxlen=90)  # 3 segundos para bostezos
        self.drowsiness_history = deque(maxlen=150)  # 5 segundos de historial
        self.closed_eye_duration = 0
        self.eye_closed_start = None
        
        # PERCLOS calculation
        self.perclos_window = deque(maxlen=1800)  # 60 segundos a 30fps
        
    def init_audio_system(self):
        """Inicializa el sistema de audio pygame"""
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # Verificar si existe el archivo de audio
            if os.path.exists("alerta.mp3"):
                self.alert_sound = pygame.mixer.Sound("alerta.mp3")
                print("✓ Audio de alerta cargado correctamente")
            else:
                print("⚠️ Archivo 'alerta.mp3' no encontrado. Creando sonido de prueba...")
                self.create_test_sound()
                
        except Exception as e:
            print(f"Error inicializando audio: {e}")
            self.alert_sound = None
    
    def create_test_sound(self):
        """Crea un sonido de alerta de prueba si no existe alerta.mp3"""
        try:
            # Generar un tono suave de alerta
            duration = 1.5  # segundos (más corto)
            sample_rate = 22050
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Combinar varias frecuencias para un sonido agradable pero efectivo
            wave = (np.sin(2 * np.pi * 440 * t) * 0.4 +  # La4
                   np.sin(2 * np.pi * 554.37 * t) * 0.3 +  # Do#5
                   np.sin(2 * np.pi * 659.25 * t) * 0.2)   # Mi5
            
            # Aplicar envelope para suavizar (más gradual)
            envelope = np.exp(-t * 0.8)
            wave = wave * envelope
            
            # Convertir a formato pygame
            wave = (wave * 32767 * 0.7).astype(np.int16)  # Volumen más bajo
            stereo_wave = np.array([wave, wave]).T
            
            self.alert_sound = pygame.sndarray.make_sound(stereo_wave)
            print("✓ Sonido de alerta de prueba generado")
            
        except Exception as e:
            print(f"Error creando sonido de prueba: {e}")
            self.alert_sound = None
    
    def play_gradual_alert(self, alert_level):
        """Reproduce alerta de forma gradual según el nivel"""
        if self.alert_sound is None or self.is_playing_alert:
            return
        
        current_time = time.time()
        
        # Evitar alertas muy frecuentes del mismo nivel
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        self.is_playing_alert = True
        self.last_alert_time = current_time
        
        # Configurar parámetros según el nivel (más suaves)
        if alert_level == 1:
            volume = 0.2
            repetitions = 1
            interval = 0
        elif alert_level == 2:
            volume = 0.4
            repetitions = 2
            interval = 0.8
        else:  # alert_level == 3
            volume = 0.6
            repetitions = 3
            interval = 0.5
        
        # Ejecutar en hilo separado para no bloquear detección
        self.alert_thread = threading.Thread(
            target=self._play_alert_sequence,
            args=(volume, repetitions, interval),
            daemon=True
        )
        self.alert_thread.start()
    
    def _play_alert_sequence(self, volume, repetitions, interval):
        """Ejecuta la secuencia de alerta en hilo separado"""
        try:
            self.alert_sound.set_volume(volume)
            
            for i in range(repetitions):
                if not self.is_playing_alert:  # Permitir cancelación
                    break
                    
                self.alert_sound.play()
                
                if i < repetitions - 1:  # No esperar después del último
                    time.sleep(interval + 0.8)  # Duración base del sonido más corta
            
        except Exception as e:
            print(f"Error reproduciendo alerta: {e}")
        finally:
            time.sleep(0.5)  # Pequeña pausa antes de permitir nueva alerta
            self.is_playing_alert = False
    
    def stop_alert(self):
        """Detiene la alerta actual"""
        self.is_playing_alert = False
        if pygame.mixer.get_init():
            pygame.mixer.stop()
    
    def calculate_ear(self, landmarks, eye_points):
        """Calcula la relación de aspecto del ojo usando landmarks de MediaPipe (ORIGINAL)"""
        try:
            # Convertir landmarks a array numpy
            points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_points])
            
            # Calcular distancias
            vertical_1 = dist.euclidean(points[1], points[5])
            vertical_2 = dist.euclidean(points[2], points[4])
            horizontal = dist.euclidean(points[0], points[3])
            
            if horizontal == 0:
                return 0.3
            
            # EAR
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
        except:
            return 0.3
    
    def calculate_mar(self, landmarks):
        """Calcula la relación de aspecto de la boca (ORIGINAL)"""
        try:
            # Puntos clave de la boca para MAR
            mouth_points = [61, 291, 39, 181, 0, 17, 18, 200]  # Puntos externos de la boca
            
            points = np.array([[landmarks[i].x, landmarks[i].y] for i in mouth_points])
            
            # Distancias verticales
            vertical_1 = dist.euclidean(points[2], points[6])  # Superior-inferior centro
            vertical_2 = dist.euclidean(points[3], points[7])  # Superior-inferior lateral
            
            # Distancia horizontal
            horizontal = dist.euclidean(points[0], points[1])  # Esquinas de la boca
            
            if horizontal == 0:
                return 0.0
            
            # MAR
            mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return mar
        except:
            return 0.0
    
    def detect_mouth_opening_simple(self, landmarks, frame_height, frame_width):
        """Método alternativo más simple para detectar apertura de boca (ORIGINAL)"""
        try:
            # Puntos de la boca
            upper_lip = landmarks[13]  # Labio superior centro
            lower_lip = landmarks[14]  # Labio inferior centro
            left_corner = landmarks[61]  # Esquina izquierda
            right_corner = landmarks[291]  # Esquina derecha
            
            # Convertir coordenadas normalizadas a píxeles
            upper_y = upper_lip.y * frame_height
            lower_y = lower_lip.y * frame_height
            left_x = left_corner.x * frame_width
            right_x = right_corner.x * frame_width
            
            # Calcular apertura vertical y ancho horizontal
            mouth_height = abs(lower_y - upper_y)
            mouth_width = abs(right_x - left_x)
            
            # Relación altura/ancho
            if mouth_width > 0:
                mouth_ratio = mouth_height / mouth_width
                return mouth_ratio
            return 0
        except:
            return 0
    
    def calculate_perclos(self):
        """Calcula el porcentaje de tiempo con ojos cerrados"""
        if len(self.perclos_window) == 0:
            return 0.0
        
        closed_frames = sum(1 for ear in self.perclos_window if ear < self.EYE_AR_THRESH)
        return closed_frames / len(self.perclos_window)
    
    def detect_microsleep(self, current_time):
        """Detecta episodios de microsueño"""
        if self.eye_closed_start is not None:
            duration = current_time - self.eye_closed_start
            if duration > 0.5:  # 0.5 segundos
                return True, duration
        return False, 0
    
    def calculate_simple_drowsiness_score(self, avg_ear, mouth_ratio, perclos, microsleep_duration):
        """Calcula un score de somnolencia simplificado pero efectivo"""
        score = 0
        
        # Ojos cerrados (peso alto)
        if avg_ear < self.EYE_AR_THRESH:
            score += 40
        elif avg_ear < 0.28:  # Ojos semi-cerrados
            score += 20
        
        # PERCLOS (peso medio)
        score += perclos * 30
        
        # Bostezos (peso medio)
        if mouth_ratio > self.MOUTH_AR_THRESH:
            score += 25
        
        # Microsueño (peso alto)
        if microsleep_duration > 0:
            score += min(microsleep_duration * 20, 40)
        
        return min(score, 100)
    
    def check_drowsiness_alerts(self):
        """Evalúa si debe activar alertas basado en el score de somnolencia"""
        # Agregar score actual al historial
        self.drowsiness_history.append(self.drowsiness_score)
        
        # Calcular promedio de los últimos 5 segundos para evitar falsos positivos
        avg_score = np.mean(list(self.drowsiness_history)) if self.drowsiness_history else 0
        
        # Determinar nivel de alerta (más conservador)
        new_alert_level = 0
        if avg_score >= self.ALERT_LEVEL_3:
            new_alert_level = 3
        elif avg_score >= self.ALERT_LEVEL_2:
            new_alert_level = 2
        elif avg_score >= self.ALERT_LEVEL_1:
            new_alert_level = 1
        
        # Activar alerta solo si es sostenido
        if new_alert_level > 0 and new_alert_level >= self.current_alert_level:
            self.play_gradual_alert(new_alert_level)
            if new_alert_level != self.current_alert_level:
                print(f"🚨 ALERTA NIVEL {new_alert_level} - Score: {avg_score:.1f}%")
        
        self.current_alert_level = new_alert_level
        return new_alert_level, avg_score
    
    def process_frame(self, frame):
        """Procesa el frame para detectar bostezos y estado de ojos (COMBINANDO ORIGINAL + ALERTAS)"""
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        current_time = time.time()
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Calcular EAR para ambos ojos (ORIGINAL)
                left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_EAR)
                right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_EAR)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Agregar al historial
                self.ear_history.append(avg_ear)
                self.perclos_window.append(avg_ear)
                
                # Calcular apertura de boca (método simple ORIGINAL)
                mouth_ratio = self.detect_mouth_opening_simple(landmarks, frame_height, frame_width)
                self.mouth_history.append(mouth_ratio)
                
                # DIBUJAR TODOS LOS LANDMARKS COMO EN EL ORIGINAL
                # Dibujar landmarks de ojos
                for idx in self.LEFT_EYE_LANDMARKS:
                    x = int(landmarks[idx].x * frame_width)
                    y = int(landmarks[idx].y * frame_height)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                for idx in self.RIGHT_EYE_LANDMARKS:
                    x = int(landmarks[idx].x * frame_width)
                    y = int(landmarks[idx].y * frame_height)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # Dibujar TODOS los landmarks de boca con diferentes colores (ORIGINAL)
                # Puntos exteriores en amarillo
                for idx in self.MOUTH_OUTER_LANDMARKS:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * frame_width)
                        y = int(landmarks[idx].y * frame_height)
                        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                
                # Puntos interiores en cyan
                for idx in self.MOUTH_INNER_LANDMARKS:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * frame_width)
                        y = int(landmarks[idx].y * frame_height)
                        cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
                
                # Puntos clave en rojo (más grandes)
                for idx in self.MOUTH_KEY_POINTS:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * frame_width)
                        y = int(landmarks[idx].y * frame_height)
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                
                # Puntos adicionales en magenta
                additional_mouth_points = [i for i in self.ALL_MOUTH_LANDMARKS 
                                         if i not in self.MOUTH_OUTER_LANDMARKS 
                                         and i not in self.MOUTH_INNER_LANDMARKS 
                                         and i not in self.MOUTH_KEY_POINTS]
                
                for idx in additional_mouth_points:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * frame_width)
                        y = int(landmarks[idx].y * frame_height)
                        cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)
                
                # DETECCIÓN COMO EN EL ORIGINAL
                # Detección de estado de ojos
                if avg_ear < self.EYE_AR_THRESH:
                    if self.eye_closed_start is None:
                        self.eye_closed_start = current_time
                    self.eye_counter += 1
                    cv2.putText(frame, "OJOS CERRADOS", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if self.eye_counter >= self.EYE_AR_CONSEC_FRAMES:
                        self.total_blinks += 1
                    self.eye_counter = 0
                    self.eye_closed_start = None
                    cv2.putText(frame, "OJOS ABIERTOS", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Detectar microsueño
                microsleep, microsleep_duration = self.detect_microsleep(current_time)
                if microsleep:
                    cv2.putText(frame, f"MICROSUEÑO: {microsleep_duration:.1f}s", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Detección de bostezos (ORIGINAL)
                if mouth_ratio > self.MOUTH_AR_THRESH:
                    cv2.putText(frame, "BOSTEZO DETECTADO!", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not self.yawn_detected:
                        self.total_yawns += 1
                        self.yawn_detected = True
                else:
                    self.yawn_detected = False
                
                # Calcular PERCLOS
                perclos = self.calculate_perclos()
                
                # Calcular score simplificado
                self.drowsiness_score = self.calculate_simple_drowsiness_score(
                    avg_ear, mouth_ratio, perclos, microsleep_duration if microsleep else 0
                )
                
                # Evaluar alertas
                alert_level, avg_drowsiness = self.check_drowsiness_alerts()
                
                # Mostrar score de somnolencia
                score_color = (0, 255, 0) if self.drowsiness_score < 40 else (0, 165, 255) if self.drowsiness_score < 70 else (0, 0, 255)
                cv2.putText(frame, f"SOMNOLENCIA: {self.drowsiness_score:.1f}%", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2)
                
                # Indicador de alerta activa
                if alert_level > 0:
                    alert_colors = [(0, 255, 255), (0, 165, 255), (0, 0, 255)]  # Amarillo, Naranja, Rojo
                    alert_texts = ["PRECAUCIÓN", "ALERTA", "PELIGRO"]
                    
                    cv2.rectangle(frame, (10, 170), (350, 200), alert_colors[alert_level-1], -1)
                    cv2.putText(frame, f"🚨 {alert_texts[alert_level-1]} - NIVEL {alert_level}", 
                               (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Indicador de audio
                if self.is_playing_alert:
                    cv2.putText(frame, "♪ REPRODUCIENDO ALERTA ♪", (10, 220),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Mostrar métricas (ORIGINAL)
                cv2.putText(frame, f"EAR: {avg_ear:.3f}", (300, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"MAR: {mouth_ratio:.3f}", (300, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"PERCLOS: {perclos:.3f}", (300, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Parpadeos: {self.total_blinks}", (10, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Bostezos: {self.total_yawns}", (10, 270),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Leyenda de colores para los puntos de la boca (ORIGINAL)
                cv2.putText(frame, "Boca - Amarillo: Exterior, Cyan: Interior", (10, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, "Rojo: Puntos clave, Magenta: Adicionales", (10, 320),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            # No se detectó rostro
            cv2.putText(frame, "NO SE DETECTA ROSTRO", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def run_detection(self):
        """Ejecuta la detección en tiempo real"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: No se pudo acceder a la cámara")
            return
        
        print("Detector de Bostezos y Estado de Ojos con Alertas de Audio")
        print("Características:")
        print("- Detección precisa con TODOS los landmarks originales")
        print("- PERCLOS (porcentaje ojos cerrados)")
        print("- Detección de microsueño")
        print("- Score combinado de somnolencia")
        print("- Alertas de audio graduales y suaves")
        print("\nNiveles de Alerta:")
        print("- Nivel 1 (50%+): Alerta suave (1 sonido, volumen muy bajo)")
        print("- Nivel 2 (70%+): Alerta media (2 sonidos, volumen medio)")
        print("- Nivel 3 (85%+): Alerta fuerte (3 sonidos, volumen moderado)")
        print("\nControles:")
        print("- 'q': Salir")
        print("- 'r': Reiniciar contadores")
        print("- 's': Detener alerta actual")
        print("- 't': Reproducir alerta de prueba")
        print("\nColores de puntos de boca:")
        print("- Amarillo: Contorno exterior")
        print("- Cyan: Contorno interior") 
        print("- Rojo: Puntos clave (esquinas, centro)")
        print("- Magenta: Puntos adicionales")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer el frame")
                break
            
            # Voltear frame horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Procesar frame
            frame = self.process_frame(frame)
            
            # Mostrar instrucciones
            cv2.putText(frame, "Presiona 'q' para salir, 'r' para reiniciar, 's' para silenciar", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Mostrar frame
            cv2.imshow('Detector de Bostezos y Estado de Ojos con Alertas', frame)
            
            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.stop_alert()
                self.total_blinks = 0
                self.total_yawns = 0
                self.eye_counter = 0
                self.yawn_counter = 0
                self.yawn_detected = False
                self.current_alert_level = 0
                self.drowsiness_score = 0
                print("Contadores reiniciados")
            elif key == ord('s'):
                self.stop_alert()
                print("Alerta detenida")
            elif key == ord('t'):
                self.play_gradual_alert(2)  # Prueba con nivel 2
                print("Reproduciendo alerta de prueba")
        
        # Limpiar recursos de audio
        self.stop_alert()
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    print("Detector de Somnolencia con Máxima Precisión + Alertas de Audio")
    print("Usando TODOS los landmarks originales para máxima precisión")
    detector = EnhancedDrowsinessDetector()
    detector.run_detection()
    
if __name__ == "__main__":
    main()