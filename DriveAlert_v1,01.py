import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import time

class YawnEyeDetectorNoLib:
    def __init__(self):
        # Inicializar MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Índices de landmarks para MediaPipe Face Mesh
        # Ojos (usando landmarks específicos de MediaPipe)
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Puntos específicos para cálculo EAR
        self.LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]  # Equivalentes a dlib
        self.RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
        
        # Boca - Puntos más completos para mejor visualización
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
        
        # Todos los puntos de boca combinados para visualización
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
        
        # Umbrales
        self.EYE_AR_THRESH = 0.25
        self.MOUTH_AR_THRESH = 0.7  # Umbral para detección de boca abierta
        self.EYE_AR_CONSEC_FRAMES = 3
        
        # Contadores
        self.eye_counter = 0
        self.yawn_counter = 0
        self.total_blinks = 0
        self.total_yawns = 0
        self.yawn_detected = False
        
    def calculate_ear(self, landmarks, eye_points):
        """Calcula la relación de aspecto del ojo usando landmarks de MediaPipe"""
        # Convertir landmarks a array numpy
        points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_points])
        
        # Calcular distancias
        vertical_1 = dist.euclidean(points[1], points[5])
        vertical_2 = dist.euclidean(points[2], points[4])
        horizontal = dist.euclidean(points[0], points[3])
        
        # EAR
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def calculate_mar(self, landmarks):
        """Calcula la relación de aspecto de la boca"""
        # Puntos clave de la boca para MAR
        mouth_points = [61, 291, 39, 181, 0, 17, 18, 200]  # Puntos externos de la boca
        
        points = np.array([[landmarks[i].x, landmarks[i].y] for i in mouth_points])
        
        # Distancias verticales
        vertical_1 = dist.euclidean(points[2], points[6])  # Superior-inferior centro
        vertical_2 = dist.euclidean(points[3], points[7])  # Superior-inferior lateral
        
        # Distancia horizontal
        horizontal = dist.euclidean(points[0], points[1])  # Esquinas de la boca
        
        # MAR
        mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return mar
    
    def detect_mouth_opening_simple(self, landmarks, frame_height, frame_width):
        """Método alternativo más simple para detectar apertura de boca"""
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
    
    def process_frame(self, frame):
        """Procesa el frame para detectar bostezos y estado de ojos"""
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Calcular EAR para ambos ojos
                left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_EAR)
                right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_EAR)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Calcular apertura de boca (método simple)
                mouth_ratio = self.detect_mouth_opening_simple(landmarks, frame_height, frame_width)
                
                # Dibujar landmarks de ojos
                for idx in self.LEFT_EYE_LANDMARKS:
                    x = int(landmarks[idx].x * frame_width)
                    y = int(landmarks[idx].y * frame_height)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                for idx in self.RIGHT_EYE_LANDMARKS:
                    x = int(landmarks[idx].x * frame_width)
                    y = int(landmarks[idx].y * frame_height)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # Dibujar TODOS los landmarks de boca con diferentes colores
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
                
                # Detección de estado de ojos
                if avg_ear < self.EYE_AR_THRESH:
                    self.eye_counter += 1
                    cv2.putText(frame, "OJOS CERRADOS", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if self.eye_counter >= self.EYE_AR_CONSEC_FRAMES:
                        self.total_blinks += 1
                    self.eye_counter = 0
                    cv2.putText(frame, "OJOS ABIERTOS", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Detección de bostezos
                if mouth_ratio > self.MOUTH_AR_THRESH:
                    cv2.putText(frame, "BOSTEZO DETECTADO!", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not self.yawn_detected:
                        self.total_yawns += 1
                        self.yawn_detected = True
                else:
                    self.yawn_detected = False
                
                # Mostrar métricas
                cv2.putText(frame, f"EAR: {avg_ear:.3f}", (300, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"MAR: {mouth_ratio:.3f}", (300, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Parpadeos: {self.total_blinks}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Bostezos: {self.total_yawns}", (10, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Leyenda de colores para los puntos de la boca
                cv2.putText(frame, "Boca - Amarillo: Exterior, Cyan: Interior", (10, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, "Rojo: Puntos clave, Magenta: Adicionales", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def run_detection(self):
        """Ejecuta la detección en tiempo real"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: No se pudo acceder a la cámara")
            return
        
        print("Detector de Bostezos y Estado de Ojos")
        print("Presiona 'q' para salir")
        print("Presiona 'r' para reiniciar contadores")
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
            cv2.putText(frame, "Presiona 'q' para salir, 'r' para reiniciar", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Mostrar frame
            cv2.imshow('Detector de Bostezos y Estado de Ojos', frame)
            
            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.total_blinks = 0
                self.total_yawns = 0
                print("Contadores reiniciados")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    print("Usando Mediapipe con puntos de boca extendidos") 
    detector = YawnEyeDetectorNoLib()
    detector.run_detection()
    
if __name__ == "__main__":
    main()