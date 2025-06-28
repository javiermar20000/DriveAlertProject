# VERSIÓN MODIFICADA PARA USAR CÁMARA DE TELÉFONO VÍA IP WEBCAM
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import time
from collections import deque
import pygame
import threading
import os

class DriverAlertTelefono:
    def __init__(self, ip_telefono="192.168.1.100", puerto="8080"):
        # URL de la cámara del teléfono (IP Webcam para Android)
        self.url_telefono = f"http://{ip_telefono}:{puerto}/video"
        
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
        self.iniciar_sistema_audio()
        
        # USAR EXACTAMENTE LOS MISMOS LANDMARKS DEL CÓDIGO ORIGINAL
        # Índices de landmarks para MediaPipe Face Mesh
        # Ojos (usando landmarks específicos de MediaPipe)
        self.PUNTOS_OJO_IZQ = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.PUNTOS_OJO_DER = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Puntos específicos para cálculo EAR (EXACTOS DEL ORIGINAL)
        self.PUNTOS_EAR_IZQ = [33, 160, 158, 133, 153, 144]  # Equivalentes a dlib
        self.PUNTOS_EAR_DER = [362, 385, 387, 263, 373, 380]
        
        # Boca - Puntos más completos para mejor visualización (EXACTOS DEL ORIGINAL)
        # Contorno exterior completo de la boca
        self.PUNTOS_BOCA_EXTERIOR = [
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            291, 303, 267, 269, 270, 271, 272, 302, 268, 271, 272
        ]
        
        # Contorno interior de la boca (labios)
        self.PUNTOS_BOCA_INTERIOR = [
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
            310, 311, 312, 13, 82, 81, 80, 78
        ]
        
        # Puntos específicos para detección (esquinas y centro)
        self.PUNTOS_BOCA_CLAVE = [
            61, 291,    # Esquinas izquierda y derecha
            13, 14,     # Centro superior e inferior
            78, 308,    # Labio superior centro
            87, 317,    # Puntos laterales superiores
            84, 314,    # Puntos externos superiores
            17, 18,     # Centro muy superior e inferior
            200, 199    # Puntos adicionales centro
        ]
        
        # Todos los puntos de boca combinados para visualización (EXACTOS DEL ORIGINAL)
        self.TODOS_PUNTOS_BOCA = list(set(
            self.PUNTOS_BOCA_EXTERIOR + 
            self.PUNTOS_BOCA_INTERIOR + 
            self.PUNTOS_BOCA_CLAVE + 
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
        
        self.PUNTOS_MAR_BOCA = [13, 14, 12, 15, 268, 269, 271, 272]  # Puntos para MAR
        
        # Umbrales (USAR LOS MISMOS DEL CÓDIGO ORIGINAL)
        self.UMBRAL_OJOS = 0.25
        self.UMBRAL_BOCA = 0.5  # Usar el umbral original más bajo
        self.FRAMES_CONSEC_OJOS = 3
        
        # Umbrales para alertas (más conservadores para reducir falsos positivos)
        self.NIVEL_ALERTA_1 = 50  # Alerta suave
        self.NIVEL_ALERTA_2 = 70  # Alerta media
        self.NIVEL_ALERTA_3 = 85  # Alerta fuerte
        
        # Contadores (EXACTOS DEL ORIGINAL)
        self.contador_ojos = 0
        self.contador_bostezos = 0
        self.total_parpadeos = 0
        self.total_bostezos = 0
        self.bostezo_detectado = False
        
        # Sistema de alertas mejorado para evitar solapamientos
        self.nivel_alerta_actual = 0
        self.tiempo_inicio_alerta = None
        self.tiempo_ultima_alerta = 0
        self.tiempo_enfriamiento = 12  # Aumentado para evitar solapamientos
        self.reproduciendo_alerta = False
        self.hilo_alerta = None
        
        # Control de duración de alertas
        self.duracion_alertas = {
            1: 3.0,   # Nivel 1: 3 segundos
            2: 4.5,   # Nivel 2: 4.5 segundos  
            3: 6.0    # Nivel 3: 6 segundos
        }
        
        # Prevenir alertas consecutivas del mismo nivel
        self.tiempo_ultima_alerta_nivel = {1: 0, 2: 0, 3: 0}
        self.enfriamiento_nivel = 15  # 15 segundos entre alertas del mismo nivel
        
        # Historial para análisis temporal (más conservador)
        self.historial_ear = deque(maxlen=150)  # 5 segundos a 30fps
        self.historial_boca = deque(maxlen=90)  # 3 segundos para bostezos
        self.historial_somnolencia = deque(maxlen=150)  # 5 segundos de historial
        self.duracion_ojos_cerrados = 0
        self.inicio_ojos_cerrados = None
        
        # PERCLOS calculation
        self.ventana_perclos = deque(maxlen=1800)  # 60 segundos a 30fps
        
    def iniciar_sistema_audio(self):
        """Inicializa el sistema de audio pygame"""
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # Verificar si existe el archivo de audio
            if os.path.exists("alerta.mp3"):
                self.sonido_alerta = pygame.mixer.Sound("alerta.mp3")
                print("✓ Audio de alerta cargado correctamente")
            else:
                print("⚠️ Archivo 'alerta.mp3' no encontrado. Creando sonido de prueba...")
                self.crear_sonido_prueba()
                
        except Exception as e:
            print(f"Error inicializando audio: {e}")
            self.sonido_alerta = None
    
    def reproducir_alerta_gradual(self, nivel_alerta):
        """Reproduce alerta de forma gradual según el nivel con mejor control temporal"""
        if self.sonido_alerta is None:
            return
        
        tiempo_actual = time.time()
        
        # Verificar si ya hay una alerta reproduciéndose
        if self.reproduciendo_alerta:
            return
        
        # Verificar cooldown general
        if tiempo_actual - self.tiempo_ultima_alerta < self.tiempo_enfriamiento:
            return
        
        # Verificar cooldown específico del nivel
        if tiempo_actual - self.tiempo_ultima_alerta_nivel[nivel_alerta] < self.enfriamiento_nivel:
            return
        
        # Iniciar nueva alerta
        self.reproduciendo_alerta = True
        self.tiempo_ultima_alerta = tiempo_actual
        self.tiempo_ultima_alerta_nivel[nivel_alerta] = tiempo_actual
        
        # Configurar parámetros según el nivel (más conservadores)
        if nivel_alerta == 1:
            volumen_inicial = 0.1
            volumen_final = 0.25
            repeticiones = 1
            duracion_fade = 1.5
            pausa_entre = 0.5
        elif nivel_alerta == 2:
            volumen_inicial = 0.15
            volumen_final = 0.4
            repeticiones = 2
            duracion_fade = 2.0
            pausa_entre = 1.0
        else:  # nivel_alerta == 3
            volumen_inicial = 0.2
            volumen_final = 0.6
            repeticiones = 2  # Reducido de 3 a 2
            duracion_fade = 2.5
            pausa_entre = 1.5
        
        # Ejecutar en hilo separado
        self.hilo_alerta = threading.Thread(
            target=self._ejecutar_secuencia_alerta,
            args=(nivel_alerta, volumen_inicial, volumen_final, repeticiones, duracion_fade, pausa_entre),
            daemon=True
        )
        self.hilo_alerta.start()
    
    def _ejecutar_secuencia_alerta(self, nivel_alerta, volumen_inicial, volumen_final, repeticiones, duracion_fade, pausa_entre):
        """Ejecuta la secuencia de alerta con duración controlada"""
        try:
            duracion_total = self.duracion_alertas[nivel_alerta]
            pasos = 15  # Reducido para menos pasos
            duracion_paso = duracion_fade / pasos
            incremento_volumen = (volumen_final - volumen_inicial) / pasos
            
            tiempo_inicio = time.time()
            
            for rep in range(repeticiones):
                # Verificar si debe detenerse por tiempo límite
                if time.time() - tiempo_inicio >= duracion_total:
                    break
                
                if not self.reproduciendo_alerta:
                    break
                
                # Fade-in más rápido
                for paso in range(pasos):
                    if not self.reproduciendo_alerta or (time.time() - tiempo_inicio >= duracion_total):
                        break
                    
                    volumen_actual = volumen_inicial + (incremento_volumen * paso)
                    self.sonido_alerta.set_volume(volumen_actual)
                    self.sonido_alerta.play()
                    time.sleep(duracion_paso)
                
                # Pausa entre repeticiones (solo si no es la última)
                if rep < repeticiones - 1 and (time.time() - tiempo_inicio < duracion_total - 1):
                    time.sleep(pausa_entre)
            
            # Fade-out rápido
            pasos_fade_out = 8
            for paso in range(pasos_fade_out):
                if not self.reproduciendo_alerta:
                    break
                
                vol_fade = volumen_final * (1.0 - paso / pasos_fade_out)
                self.sonido_alerta.set_volume(vol_fade)
                self.sonido_alerta.play()
                time.sleep(0.1)
                    
        except Exception as e:
            print(f"Error reproduciendo alerta: {e}")
        finally:
            # Asegurar que la alerta se marca como terminada
            time.sleep(0.3)
            self.reproduciendo_alerta = False
    
    def crear_sonido_prueba(self):
        """Crea un sonido más corto y menos agresivo"""
        try:
            duracion = 0.5  # Más corto
            frecuencia_muestreo = 22050
            t = np.linspace(0, duracion, int(frecuencia_muestreo * duracion))
            
            # Frecuencias más suaves
            onda = (np.sin(2 * np.pi * 200 * t) * 0.4 +      # Más grave
                   np.sin(2 * np.pi * 300 * t) * 0.3 +      
                   np.sin(2 * np.pi * 400 * t) * 0.25)      
            
            # Envelope suave
            fade_in = np.linspace(0, 1, int(frecuencia_muestreo * 0.05))  # Fade-in muy rápido
            fade_out = np.linspace(1, 0, int(frecuencia_muestreo * 0.2))   # Fade-out más largo
            
            onda[:len(fade_in)] *= fade_in
            onda[-len(fade_out):] *= fade_out
            
            # Volumen más bajo
            onda = (onda * 32767 * 0.4).astype(np.int16)
            onda_estereo = np.array([onda, onda]).T
            
            self.sonido_alerta = pygame.sndarray.make_sound(onda_estereo)
            print("✓ Sonido de alerta optimizado generado")
            
        except Exception as e:
            print(f"Error creando sonido: {e}")
            self.sonido_alerta = None

    def detener_alerta(self):
        """Detiene la alerta actual de forma inmediata"""
        self.reproduciendo_alerta = False
        if self.hilo_alerta and self.hilo_alerta.is_alive():
            # La bandera reproduciendo_alerta hará que el hilo termine
            pass
        
        # Detener cualquier sonido que esté reproduciéndose
        if pygame.mixer.get_init():
            pygame.mixer.stop()
    
    def calcular_ear(self, landmarks, puntos_ojo):
        """Calcula la relación de aspecto del ojo usando landmarks de MediaPipe (ORIGINAL)"""
        try:
            # Convertir landmarks a array numpy
            points = np.array([[landmarks[i].x, landmarks[i].y] for i in puntos_ojo])
            
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
    
    def calcular_mar(self, landmarks):
        """Calcula la relación de aspecto de la boca (ORIGINAL)"""
        try:
            # Puntos clave de la boca para MAR
            puntos_boca = [61, 291, 39, 181, 0, 17, 18, 200]  # Puntos externos de la boca
            
            points = np.array([[landmarks[i].x, landmarks[i].y] for i in puntos_boca])
            
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
    
    def detectar_apertura_boca(self, landmarks, alto_frame, ancho_frame):
        """Método alternativo más simple para detectar apertura de boca (ORIGINAL)"""
        try:
            # Puntos de la boca
            labio_superior = landmarks[13]  # Labio superior centro
            labio_inferior = landmarks[14]  # Labio inferior centro
            esquina_izq = landmarks[61]  # Esquina izquierda
            esquina_der = landmarks[291]  # Esquina derecha
            
            # Convertir coordenadas normalizadas a píxeles
            superior_y = labio_superior.y * alto_frame
            inferior_y = labio_inferior.y * alto_frame
            izquierda_x = esquina_izq.x * ancho_frame
            derecha_x = esquina_der.x * ancho_frame
            
            # Calcular apertura vertical y ancho horizontal
            altura_boca = abs(inferior_y - superior_y)
            ancho_boca = abs(derecha_x - izquierda_x)
            
            # Relación altura/ancho
            if ancho_boca > 0:
                ratio_boca = altura_boca / ancho_boca
                return ratio_boca
            return 0
        except:
            return 0
    
    def calcular_perclos(self):
        """Calcula el porcentaje de tiempo con ojos cerrados"""
        if len(self.ventana_perclos) == 0:
            return 0.0
        
        frames_cerrados = sum(1 for ear in self.ventana_perclos if ear < self.UMBRAL_OJOS)
        return frames_cerrados / len(self.ventana_perclos)
    
    def detectar_microsueno(self, tiempo_actual):
        """Detecta episodios de microsueño"""
        if self.inicio_ojos_cerrados is not None:
            duracion = tiempo_actual - self.inicio_ojos_cerrados
            if duracion > 0.5:  # 0.5 segundos
                return True, duracion
        return False, 0
    
    def calcular_puntaje_somnolencia(self, ear_promedio, ratio_boca, perclos, duracion_microsueno):
        """Calcula un puntaje de somnolencia simplificado pero efectivo"""
        puntaje = 0
        
        # Ojos cerrados (peso alto)
        if ear_promedio < self.UMBRAL_OJOS:
            puntaje += 40
        elif ear_promedio < 0.28:  # Ojos semi-cerrados
            puntaje += 20
        
        # PERCLOS (peso medio)
        puntaje += perclos * 30
        
        # Bostezos (peso medio)
        if ratio_boca > self.UMBRAL_BOCA:
            puntaje += 25
        
        # Microsueño (peso alto)
        if duracion_microsueno > 0:
            puntaje += min(duracion_microsueno * 20, 40)
        
        return min(puntaje, 100)
    
    def verificar_alertas_somnolencia(self):
        """Sistema de alertas con mejor control temporal"""
        self.historial_somnolencia.append(self.puntaje_somnolencia)
        
        # Suavizar con ventana más grande para evitar alertas erráticas
        puntajes_recientes = list(self.historial_somnolencia)[-120:]  # 4 segundos a 30fps
        puntaje_promedio = np.mean(puntajes_recientes) if puntajes_recientes else 0
        
        # Umbrales más conservadores para evitar alertas excesivas
        nuevo_nivel_alerta = 0
        if puntaje_promedio >= 85:  # Más alto para nivel 3
            nuevo_nivel_alerta = 3
        elif puntaje_promedio >= 70:  # Más alto para nivel 2
            nuevo_nivel_alerta = 2
        elif puntaje_promedio >= 50:  # Más alto para nivel 1
            nuevo_nivel_alerta = 1
        
        tiempo_actual = time.time()
        
        # Lógica mejorada para evitar solapamientos
        debe_alertar = False
        
        # Solo alertar si:
        # 1. El nivel ha aumentado significativamente
        # 2. Ha pasado suficiente tiempo desde la última alerta
        # 3. No hay una alerta reproduciéndose actualmente
        
        if not self.reproduciendo_alerta:
            if nuevo_nivel_alerta > self.nivel_alerta_actual:
                # Nivel aumentó - alertar inmediatamente
                debe_alertar = True
            elif nuevo_nivel_alerta > 0:
                # Mismo nivel o menor - verificar tiempos de cooldown
                tiempo_desde_ultima = tiempo_actual - self.tiempo_ultima_alerta
                tiempo_desde_nivel = tiempo_actual - self.tiempo_ultima_alerta_nivel.get(nuevo_nivel_alerta, 0)
                
                if tiempo_desde_ultima > self.tiempo_enfriamiento and tiempo_desde_nivel > self.enfriamiento_nivel:
                    debe_alertar = True
        
        if debe_alertar:
            self.reproducir_alerta_gradual(nuevo_nivel_alerta)
            mensajes_alerta = ["", "🔔 ALERTA SUAVE", "🔊 ALERTA MEDIA", "🚨 ALERTA FUERTE"]
            print(f"{mensajes_alerta[nuevo_nivel_alerta]} - Puntaje: {puntaje_promedio:.1f}%")
        
        self.nivel_alerta_actual = nuevo_nivel_alerta
        return nuevo_nivel_alerta, puntaje_promedio
    
    def procesar_frame(self, frame):
        """Procesa el frame para detectar bostezos y estado de ojos (COMBINANDO ORIGINAL + ALERTAS)"""
        alto_frame, ancho_frame = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        tiempo_actual = time.time()
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Calcular EAR para ambos ojos (ORIGINAL)
                ear_izq = self.calcular_ear(landmarks, self.PUNTOS_EAR_IZQ)
                ear_der = self.calcular_ear(landmarks, self.PUNTOS_EAR_DER)
                ear_promedio = (ear_izq + ear_der) / 2.0
                
                # Agregar al historial
                self.historial_ear.append(ear_promedio)
                self.ventana_perclos.append(ear_promedio)
                
                # Calcular apertura de boca (método simple ORIGINAL)
                ratio_boca = self.detectar_apertura_boca(landmarks, alto_frame, ancho_frame)
                self.historial_boca.append(ratio_boca)
                
                # DIBUJAR TODOS LOS LANDMARKS COMO EN EL ORIGINAL
                # Dibujar landmarks de ojos
                for idx in self.PUNTOS_OJO_IZQ:
                    x = int(landmarks[idx].x * ancho_frame)
                    y = int(landmarks[idx].y * alto_frame)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                for idx in self.PUNTOS_OJO_DER:
                    x = int(landmarks[idx].x * ancho_frame)
                    y = int(landmarks[idx].y * alto_frame)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # Dibujar TODOS los landmarks de boca con diferentes colores (ORIGINAL)
                # Puntos exteriores en amarillo
                for idx in self.PUNTOS_BOCA_EXTERIOR:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * ancho_frame)
                        y = int(landmarks[idx].y * alto_frame)
                        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                
                # Puntos interiores en cyan
                for idx in self.PUNTOS_BOCA_INTERIOR:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * ancho_frame)
                        y = int(landmarks[idx].y * alto_frame)
                        cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
                
                # Puntos clave en rojo (más grandes)
                for idx in self.PUNTOS_BOCA_CLAVE:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * ancho_frame)
                        y = int(landmarks[idx].y * alto_frame)
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                
                # Puntos adicionales en magenta
                puntos_adicionales_boca = [i for i in self.TODOS_PUNTOS_BOCA 
                                         if i not in self.PUNTOS_BOCA_EXTERIOR 
                                         and i not in self.PUNTOS_BOCA_INTERIOR 
                                         and i not in self.PUNTOS_BOCA_CLAVE]
                
                for idx in puntos_adicionales_boca:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * ancho_frame)
                        y = int(landmarks[idx].y * alto_frame)
                        cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)
                
                # DETECCIÓN COMO EN EL ORIGINAL
                # Detección de estado de ojos
                if ear_promedio < self.UMBRAL_OJOS:
                    if self.inicio_ojos_cerrados is None:
                        self.inicio_ojos_cerrados = tiempo_actual
                    self.contador_ojos += 1
                    cv2.putText(frame, "OJOS CERRADOS", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if self.contador_ojos >= self.FRAMES_CONSEC_OJOS:
                        self.total_parpadeos += 1
                    self.contador_ojos = 0
                    self.inicio_ojos_cerrados = None
                    cv2.putText(frame, "OJOS ABIERTOS", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Detectar microsueño
                microsueno, duracion_microsueno = self.detectar_microsueno(tiempo_actual)
                if microsueno:
                    cv2.putText(frame, f"MICROSUEÑO: {duracion_microsueno:.1f}s", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Detección de bostezos (ORIGINAL)
                if ratio_boca > self.UMBRAL_BOCA:
                    cv2.putText(frame, "BOSTEZO DETECTADO!", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not self.bostezo_detectado:
                        self.total_bostezos += 1
                        self.bostezo_detectado = True
                else:
                    self.bostezo_detectado = False
                
                # Calcular PERCLOS
                perclos = self.calcular_perclos()
                
                # Calcular puntaje simplificado
                self.puntaje_somnolencia = self.calcular_puntaje_somnolencia(
                    ear_promedio, ratio_boca, perclos, duracion_microsueno if microsueno else 0
                )
                
                # Evaluar alertas
                nivel_alerta, somnolencia_promedio = self.verificar_alertas_somnolencia()
                
                # Mostrar puntaje de somnolencia
                color_puntaje = (0, 255, 0) if self.puntaje_somnolencia < 40 else (0, 165, 255) if self.puntaje_somnolencia < 70 else (0, 0, 255)
                cv2.putText(frame, f"SOMNOLENCIA: {self.puntaje_somnolencia:.1f}%", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_puntaje, 2)
                
                # Indicador de alerta activa
                if nivel_alerta > 0:
                    colores_alerta = [(0, 255, 255), (0, 165, 255), (0, 0, 255)]  # Amarillo, Naranja, Rojo
                    textos_alerta = ["PRECAUCIÓN", "ALERTA", "PELIGRO"]
                    
                    cv2.rectangle(frame, (10, 170), (350, 200), colores_alerta[nivel_alerta-1], -1)
                    cv2.putText(frame, f"🚨 {textos_alerta[nivel_alerta-1]} - NIVEL {nivel_alerta}", 
                               (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Indicador de audio
                if self.reproduciendo_alerta:
                    cv2.putText(frame, "♪ REPRODUCIENDO ALERTA ♪", (10, 220),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Mostrar métricas (ORIGINAL)
                cv2.putText(frame, f"EAR: {ear_promedio:.3f}", (300, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"MAR: {ratio_boca:.3f}", (300, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"PERCLOS: {perclos:.3f}", (300, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Parpadeos: {self.total_parpadeos}", (10, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Bostezos: {self.total_bostezos}", (10, 270),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Leyenda de colores para los puntos de la boca (ORIGINAL)
                cv2.putText(frame, "Boca - Amarillo: Exterior, Cyan: Interior", (10, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, "Rojo: Puntos clave, Magenta: Adicionales", (10, 320),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Información de conexión específica para teléfono
                cv2.putText(frame, f"📱 Cámara del teléfono conectada", (10, 340),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            # No se detectó rostro
            cv2.putText(frame, "NO SE DETECTA ROSTRO", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def probar_conexion_telefono(self):
        """Prueba la conexión con la cámara del teléfono"""
        print(f"Probando conexión con: {self.url_telefono}")
        try:
            cap = cv2.VideoCapture(self.url_telefono)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                print("✅ Conexión exitosa con la cámara del teléfono!")
                return True
            else:
                print("❌ No se pudo conectar. Verifica:")
                print("1. Que IP Webcam esté ejecutándose en tu teléfono")
                print("2. Que ambos dispositivos estén en la misma red WiFi")
                print("3. Que la IP y puerto sean correctos")
                return False
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            return False
    
    def iniciar_deteccion(self):
        """Ejecuta la detección usando la cámara del teléfono con información mejorada"""
        
        # Probar conexión primero
        if not self.probar_conexion_telefono():
            print("\n💡 INSTRUCCIONES:")
            print("1. Instala 'IP Webcam' desde Play Store")
            print("2. Abre la app y presiona 'Start server'")
            print("3. Anota la IP que aparece (ej: 192.168.1.100:8080)")
            print("4. Ejecuta este script con: detector = DriverAlertTelefono('TU_IP', 'PUERTO')")
            return
        
        cap = cv2.VideoCapture(self.url_telefono)
        
        if not cap.isOpened():
            print("Error: No se pudo conectar a la cámara del teléfono")
            return
        
        print("\nControles:")
        print("- 'q': Salir")
        print("- 'r': Reiniciar contadores")
        print("- 's': Detener alerta")
        print("- 't': Prueba de audio")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer el frame de la cámara del teléfono")
                break
            
            # No voltear porque ya viene desde el teléfono en orientación correcta
            frame = self.procesar_frame(frame)
            
            # Mostrar instrucciones
            cv2.putText(frame, "Presiona 'q' para salir, 'r' para reiniciar, 's' para silenciar", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow('Detector de Somnolencia - Cámara de Teléfono', frame)
            
            # Controles de teclado (IGUALES A MAIN.PY)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.detener_alerta()
                self.total_parpadeos = 0
                self.total_bostezos = 0
                self.contador_ojos = 0
                self.contador_bostezos = 0
                self.bostezo_detectado = False
                self.nivel_alerta_actual = 0
                self.puntaje_somnolencia = 0
                print("Contadores reiniciados")
            elif key == ord('s'):
                self.detener_alerta()
                print("Alerta detenida")
            elif key == ord('t'):
                self.reproducir_alerta_gradual(2)  # Prueba con nivel 2
                print("Reproduciendo alerta de prueba")
        
        # Limpiar recursos de audio
        self.detener_alerta()
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    print("🚗 Detector de Somnolencia para Cámara de Teléfono con Máxima Precisión")
    print("Usando TODOS los puntos faciales originales para máxima precisión")
    print("=" * 70)
    
    # Cambiar por la IP de tu teléfono
    ip_telefono = input("Ingresa la IP de tu teléfono (ej: 192.168.1.100): ").strip()
    if not ip_telefono:
        ip_telefono = "192.168.1.124"  # IP por defecto
    
    puerto = input("Ingresa el puerto (presiona Enter para 8080): ").strip()
    if not puerto:
        puerto = "8080"
    
    detector = DriverAlertTelefono(ip_telefono, puerto)
    detector.iniciar_deteccion()

if __name__ == "__main__":
    main()