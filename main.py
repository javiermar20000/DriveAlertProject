import cv2 #para procesamiento de imagenes y vision por computadora
import numpy as np #para manejo de arreglos y operaciones matematicas
import mediapipe as mp
from scipy.spatial import distance as dist #funcion distance de scipy para calcular distancias euclidianas
import time
from collections import deque
import pygame #para reproducir sonidos de alerta
import threading
import os

#clase principal que maneja toda la deteccion de somnolencia y sistema de alertas
class DriverAlert:
    def __init__(self):
        #inicializacion del detector de puntos faciales mediapipe
        self.mp_face_mesh = mp.solutions.face_mesh
        #configuracion del detector facial con parametros optimizados
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        #utilidades para dibujar los puntos de referencia
        self.mp_drawing = mp.solutions.drawing_utils
        
        #inicializar sistema de reproduccion de audio para alertas
        self.iniciar_sistema_audio()
        
        #indices de puntos de referencia para los ojos segun mediapipe
        self.PUNTOS_OJO_IZQ = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.PUNTOS_OJO_DER = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        #puntos especificos para calcular la relacion de aspecto del ojo
        self.PUNTOS_EAR_IZQ = [33, 160, 158, 133, 153, 144]
        self.PUNTOS_EAR_DER = [362, 385, 387, 263, 373, 380]
        
        #puntos que definen el contorno exterior de la boca
        self.PUNTOS_BOCA_EXTERIOR = [
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            291, 303, 267, 269, 270, 271, 272, 302, 268, 271, 272
        ]
        
        #puntos que definen el contorno interior de la boca
        self.PUNTOS_BOCA_INTERIOR = [
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
            310, 311, 312, 13, 82, 81, 80, 78
        ]
        
        #puntos clave para una deteccion mas precisa de la boca
        self.PUNTOS_BOCA_CLAVE = [
            61, 291,    #esquinas izquierda y derecha
            13, 14,     #centro superior e inferior
            78, 308,    #labio superior centro
            87, 317,    #puntos laterales superiores
            84, 314,    #puntos externos superiores
            17, 18,     #centro muy superior e inferior
            200, 199    #puntos adicionales centro
        ]
        
        #conjunto completo de todos los puntos de boca para visualizacion
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
        
        #puntos para calcular la relacion de aspecto de la boca
        self.PUNTOS_MAR_BOCA = [13, 14, 12, 15, 268, 269, 271, 272]
        
        #umbrales para deteccion de estados
        self.UMBRAL_OJOS = 0.25  #umbral para considerar ojo cerrado
        self.UMBRAL_BOCA = 0.5  #umbral para considerar bostezo
        self.FRAMES_CONSEC_OJOS = 3  #frames consecutivos para confirmar parpadeo
        
        #umbrales para diferentes niveles de alerta
        self.NIVEL_ALERTA_1 = 50  #umbral para alerta leve
        self.NIVEL_ALERTA_2 = 70  #umbral para alerta media
        self.NIVEL_ALERTA_3 = 85  #umbral para alerta grave
        
        #contadores y variables de estado
        self.contador_ojos = 0  #contador de frames con ojos cerrados
        self.contador_bostezos = 0  #contador para deteccion de bostezos
        self.total_parpadeos = 0  #contador total de parpadeos detectados
        self.total_bostezos = 0  #contador total de bostezos detectados
        self.bostezo_detectado = False  #bandera para control de estado de bostezo
        
        #sistema de control de alertas
        self.nivel_alerta_actual = 0  #nivel actual de alerta
        self.tiempo_inicio_alerta = None  #timestamp de inicio de la alerta actual
        self.tiempo_ultima_alerta = 0  #timestamp de la ultima alerta emitida
        self.tiempo_enfriamiento = 12  #tiempo minimo entre alertas en segundos
        self.reproduciendo_alerta = False  #bandera para indicar si hay una alerta activa
        self.hilo_alerta = None  #hilo para reproducir alertas en segundo plano
        
        #duracion de alertas segun nivel de gravedad
        self.duracion_alertas = {
            1: 3.0,   #nivel 1: alerta suave de 3 segundos
            2: 4.5,   #nivel 2: alerta media de 4.5 segundos
            3: 6.0    #nivel 3: alerta fuerte de 6 segundos
        }
        
        #control de alertas consecutivas del mismo nivel
        self.tiempo_ultima_alerta_nivel = {1: 0, 2: 0, 3: 0}  #timestamps por nivel
        self.enfriamiento_nivel = 15  #tiempo minimo entre alertas del mismo nivel
        
        #historiales temporales para analisis de patrones
        self.historial_ear = deque(maxlen=150)  #historial ear para 5 segundos a 30fps
        self.historial_boca = deque(maxlen=90)  #historial de apertura bucal
        self.historial_somnolencia = deque(maxlen=150)  #historial de somnolencia
        self.duracion_ojos_cerrados = 0  #duracion acumulada de ojos cerrados
        self.inicio_ojos_cerrados = None  #timestamp inicio de ojos cerrados
        
        #ventana para calculo de perclos (porcentaje de tiempo con ojos cerrados)
        self.ventana_perclos = deque(maxlen=1800)  #60 segundos a 30fps
        
    def iniciar_sistema_audio(self):
        #inicializa el sistema de audio pygame para reproducir alertas sonoras
        try:
            #configuracion de parametros de audio para mejor calidad y menor latencia
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            #verificacion de disponibilidad del archivo de sonido de alerta
            if os.path.exists("alerta.mp3"):
                #carga del archivo de audio si existe
                self.sonido_alerta = pygame.mixer.Sound("alerta.mp3")
                print("✓ Audio de alerta cargado correctamente")
            else:
                #generar un sonido de prueba si no se encuentra el archivo
                print("⚠️ Archivo 'alerta.mp3' no encontrado. Creando sonido de prueba...")
                self.crear_sonido_prueba()
                
        except Exception as e:
            #manejo de errores en la inicializacion del sistema de audio
            print(f"Error inicializando audio: {e}")
            self.sonido_alerta = None
    
    def reproducir_alerta_gradual(self, nivel_alerta):
        #reproduce una alerta sonora con volumen gradual segun el nivel de somnolencia
        if self.sonido_alerta is None:
            #no se reproduce nada si no hay sonido cargado
            return
        
        tiempo_actual = time.time()
        
        #evita reproducir si ya hay una alerta en curso
        if self.reproduciendo_alerta:
            return
        
        #respeta el tiempo de enfriamiento general entre alertas
        if tiempo_actual - self.tiempo_ultima_alerta < self.tiempo_enfriamiento:
            return
        
        #respeta el tiempo de enfriamiento especifico para el nivel de alerta
        if tiempo_actual - self.tiempo_ultima_alerta_nivel[nivel_alerta] < self.enfriamiento_nivel:
            return
        
        #configuracion de la nueva alerta a reproducir
        self.reproduciendo_alerta = True
        self.tiempo_ultima_alerta = tiempo_actual
        self.tiempo_ultima_alerta_nivel[nivel_alerta] = tiempo_actual
        
        #parametros de reproduccion adaptados segun el nivel de alerta
        if nivel_alerta == 1:
            #parametros para nivel 1: alerta suave
            volumen_inicial = 0.1
            volumen_final = 0.25
            repeticiones = 1
            duracion_fade = 1.5
            pausa_entre = 0.5
        elif nivel_alerta == 2:
            #parametros para nivel 2: alerta media
            volumen_inicial = 0.15
            volumen_final = 0.4
            repeticiones = 2
            duracion_fade = 2.0
            pausa_entre = 1.0
        else:  #nivel 3: alerta maxima
            #parametros para nivel 3: alerta grave
            volumen_inicial = 0.2
            volumen_final = 0.6
            repeticiones = 2  #reducido de 3 a 2 para evitar molestias excesivas
            duracion_fade = 2.5
            pausa_entre = 1.5
        
        #ejecucion de la alerta en un hilo separado para no bloquear la interfaz
        self.hilo_alerta = threading.Thread(
            target=self._ejecutar_secuencia_alerta,
            args=(nivel_alerta, volumen_inicial, volumen_final, repeticiones, duracion_fade, pausa_entre),
            daemon=True
        )
        self.hilo_alerta.start()
    
    def _ejecutar_secuencia_alerta(self, nivel_alerta, volumen_inicial, volumen_final, repeticiones, duracion_fade, pausa_entre):
        #implementacion interna de la secuencia de alerta con control preciso de duracion
        try:
            #duracion maxima para este nivel de alerta
            duracion_total = self.duracion_alertas[nivel_alerta]
            pasos = 15  #numero de pasos para el fade-in (aumento gradual de volumen)
            duracion_paso = duracion_fade / pasos
            incremento_volumen = (volumen_final - volumen_inicial) / pasos
            
            #marca de tiempo inicial para controlar duracion total
            tiempo_inicio = time.time()
            
            for rep in range(repeticiones):
                #verificacion de tiempo limite total para evitar alertas demasiado largas
                if time.time() - tiempo_inicio >= duracion_total:
                    break
                
                #verificacion de bandera de interrupcion manual
                if not self.reproduciendo_alerta:
                    break
                
                #fade-in: aumento gradual del volumen para no sobresaltar
                for paso in range(pasos):
                    #verificaciones de seguridad para permitir interrupciones
                    if not self.reproduciendo_alerta or (time.time() - tiempo_inicio >= duracion_total):
                        break
                    
                    #calcular y aplicar volumen actual
                    volumen_actual = volumen_inicial + (incremento_volumen * paso)
                    self.sonido_alerta.set_volume(volumen_actual)
                    self.sonido_alerta.play()
                    time.sleep(duracion_paso)
                
                #pausa entre repeticiones si no es la ultima
                if rep < repeticiones - 1 and (time.time() - tiempo_inicio < duracion_total - 1):
                    time.sleep(pausa_entre)
            
            #fade-out: reduccion gradual del volumen al final de la alerta
            pasos_fade_out = 8
            for paso in range(pasos_fade_out):
                #verificacion de bandera de interrupcion
                if not self.reproduciendo_alerta:
                    break
                
                #calculo y aplicacion del volumen decreciente
                vol_fade = volumen_final * (1.0 - paso / pasos_fade_out)
                self.sonido_alerta.set_volume(vol_fade)
                self.sonido_alerta.play()
                time.sleep(0.1)
                    
        except Exception as e:
            #manejo de errores durante la reproduccion
            print(f"Error reproduciendo alerta: {e}")
        finally:
            #asegurar limpieza de recursos y estados al finalizar
            time.sleep(0.3)
            self.reproduciendo_alerta = False
    
    def detener_alerta(self):
        #detiene inmediatamente cualquier alerta que se este reproduciendo
        self.reproduciendo_alerta = False
        #detener reproduccion de sonido actual si esta activa
        if pygame.mixer.get_init():
            pygame.mixer.stop()
    
    def crear_sonido_prueba(self):
        #crea un sonido de alerta sintetico cuando no se encuentra el archivo de audio
        try:
            #parametros para un sonido corto y moderado
            duracion = 0.5
            frecuencia_muestreo = 22050
            #generar array de tiempo para la onda
            t = np.linspace(0, duracion, int(frecuencia_muestreo * duracion))
            
            #crear onda usando combinacion de frecuencias para un tono de alerta
            onda = (np.sin(2 * np.pi * 200 * t) * 0.4 +  #tono grave
                   np.sin(2 * np.pi * 300 * t) * 0.3 +  #tono medio  
                   np.sin(2 * np.pi * 400 * t) * 0.25)  #tono agudo
            
            #aplicar envolvente para suavizar inicio y fin del sonido
            fade_in = np.linspace(0, 1, int(frecuencia_muestreo * 0.05))  #fade-in rapido
            fade_out = np.linspace(1, 0, int(frecuencia_muestreo * 0.2))  #fade-out mas suave
            
            #aplicar envolventes a la onda
            onda[:len(fade_in)] *= fade_in
            onda[-len(fade_out):] *= fade_out
            
            #normalizar y convertir a formato int16 requerido por pygame
            onda = (onda * 32767 * 0.4).astype(np.int16)
            #convertir a estereo para mejor calidad
            onda_estereo = np.array([onda, onda]).T
            
            #crear objeto de sonido pygame
            self.sonido_alerta = pygame.sndarray.make_sound(onda_estereo)
            print("✓ Sonido de alerta optimizado generado")
            
        except Exception as e:
            #manejo de errores durante la generacion de sonido
            print(f"Error creando sonido: {e}")
            self.sonido_alerta = None

    def calcular_ear(self, landmarks, puntos_ojo):
        #calcula la relacion de aspecto del ojo para detectar parpadeos
        try:
            #convertir los puntos del ojo a coordenadas numpy para calculos
            points = np.array([[landmarks[i].x, landmarks[i].y] for i in puntos_ojo])
            
            #calcular las distancias verticales y horizontales entre puntos clave
            vertical_1 = dist.euclidean(points[1], points[5])
            vertical_2 = dist.euclidean(points[2], points[4])
            horizontal = dist.euclidean(points[0], points[3])
            
            #evitar division por cero
            if horizontal == 0:
                return 0.3  #valor por defecto seguro
            
            #formula de ear: promedio de distancias verticales dividido por distancia horizontal
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
        except:
            #valor por defecto en caso de error
            return 0.3
    
    def calcular_mar(self, landmarks):
        #calcula la relacion de aspecto de la boca para detectar bostezos
        try:
            #puntos clave externos e internos de la boca
            puntos_boca = [61, 291, 39, 181, 0, 17, 18, 200]
            
            #convertir puntos a coordenadas numpy
            points = np.array([[landmarks[i].x, landmarks[i].y] for i in puntos_boca])
            
            #calcular distancias verticales de apertura bucal
            vertical_1 = dist.euclidean(points[2], points[6])  #centro superior-inferior
            vertical_2 = dist.euclidean(points[3], points[7])  #lateral superior-inferior
            
            #distancia horizontal entre esquinas de la boca
            horizontal = dist.euclidean(points[0], points[1])
            
            #evitar division por cero
            if horizontal == 0:
                return 0.0
            
            #formula mar: promedio de distancias verticales dividido por distancia horizontal
            mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return mar
        except:
            #valor por defecto en caso de error
            return 0.0
    
    def detectar_apertura_boca(self, landmarks, alto_frame, ancho_frame):
        #metodo alternativo para detectar apertura de boca usando cuatro puntos principales
        try:
            #puntos clave de la boca: labios superior e inferior y esquinas
            labio_superior = landmarks[13]  #punto central del labio superior
            labio_inferior = landmarks[14]  #punto central del labio inferior
            esquina_izq = landmarks[61]  #esquina izquierda de la boca
            esquina_der = landmarks[291]  #esquina derecha de la boca
            
            #convertir coordenadas normalizadas a pixeles para calculos precisos
            superior_y = labio_superior.y * alto_frame
            inferior_y = labio_inferior.y * alto_frame
            izquierda_x = esquina_izq.x * ancho_frame
            derecha_x = esquina_der.x * ancho_frame
            
            #calcular altura vertical y ancho horizontal de la boca
            altura_boca = abs(inferior_y - superior_y)
            ancho_boca = abs(derecha_x - izquierda_x)
            
            #calcular relacion altura/ancho como indicador de apertura
            if ancho_boca > 0:
                ratio_boca = altura_boca / ancho_boca
                return ratio_boca
            return 0
        except:
            #valor por defecto en caso de error
            return 0
    
    def calcular_perclos(self):
        #calcula el porcentaje de tiempo con ojos cerrados (perclos)
        if len(self.ventana_perclos) == 0:
            #si no hay datos suficientes, retornar 0
            return 0.0
        
        #contar frames con ojos cerrados y dividir por total de frames
        frames_cerrados = sum(1 for ear in self.ventana_perclos if ear < self.UMBRAL_OJOS)
        return frames_cerrados / len(self.ventana_perclos)
    
    def detectar_microsueno(self, tiempo_actual):
        #detecta episodios de microsueño cuando los ojos permanecen cerrados
        if self.inicio_ojos_cerrados is not None:
            #calcular duracion con ojos cerrados
            duracion = tiempo_actual - self.inicio_ojos_cerrados
            #umbral de 0.5 segundos para considerar microsueño
            if duracion > 0.5:
                return True, duracion
        #retornar falso si no hay microsueño detectado
        return False, 0
    
    def calcular_puntaje_somnolencia(self, ear_promedio, ratio_boca, perclos, duracion_microsueno):
        #calcula un puntaje de somnolencia combinando multiples factores
        puntaje = 0
        
        #factor 1: ojos cerrados (peso alto)
        if ear_promedio < self.UMBRAL_OJOS:
            #ojos completamente cerrados
            puntaje += 40
        elif ear_promedio < 0.28:
            #ojos semicerrados
            puntaje += 20
        
        #factor 2: perclos (peso medio) - porcentaje de tiempo con ojos cerrados
        puntaje += perclos * 30
        
        #factor 3: bostezos (peso medio)
        if ratio_boca > self.UMBRAL_BOCA:
            puntaje += 25
        
        #factor 4: microsueño (peso alto) - valora mas los microsueños largos
        if duracion_microsueno > 0:
            #puntaje proporcional a duracion con limite superior
            puntaje += min(duracion_microsueno * 20, 40)
        
        #limitar el puntaje maximo a 100
        return min(puntaje, 100)
    
    def verificar_alertas_somnolencia(self):
        #sistema inteligente de alertas con control temporal
        #guardar puntaje actual en el historial
        self.historial_somnolencia.append(self.puntaje_somnolencia)
        
        #suavizar puntaje usando promedio de ventana deslizante
        puntajes_recientes = list(self.historial_somnolencia)[-120:]  #ultimos 4 segundos a 30fps
        puntaje_promedio = np.mean(puntajes_recientes) if puntajes_recientes else 0
        
        #determinar nivel de alerta segun puntaje promedio
        nuevo_nivel_alerta = 0
        if puntaje_promedio >= 85:
            #nivel 3: somnolencia severa
            nuevo_nivel_alerta = 3
        elif puntaje_promedio >= 70:
            #nivel 2: somnolencia moderada
            nuevo_nivel_alerta = 2
        elif puntaje_promedio >= 50:
            #nivel 1: somnolencia leve
            nuevo_nivel_alerta = 1
        
        tiempo_actual = time.time()
        
        #bandera para decidir si disparar una alerta
        debe_alertar = False
        
        #logica para evitar alertas repetitivas o solapadas
        if not self.reproduciendo_alerta:
            if nuevo_nivel_alerta > self.nivel_alerta_actual:
                #alerta inmediata si el nivel empeora
                debe_alertar = True
            elif nuevo_nivel_alerta > 0:
                #verificar tiempos de enfriamiento para alertas del mismo nivel
                tiempo_desde_ultima = tiempo_actual - self.tiempo_ultima_alerta
                tiempo_desde_nivel = tiempo_actual - self.tiempo_ultima_alerta_nivel.get(nuevo_nivel_alerta, 0)
                
                #solo alertar si se respetan ambos cooldowns
                if tiempo_desde_ultima > self.tiempo_enfriamiento and tiempo_desde_nivel > self.enfriamiento_nivel:
                    debe_alertar = True
        
        #reproducir alerta si corresponde
        if debe_alertar:
            self.reproducir_alerta_gradual(nuevo_nivel_alerta)
            #mensajes diferenciados por nivel de alerta
            mensajes_alerta = ["", "🔔 ALERTA SUAVE", "🔊 ALERTA MEDIA", "🚨 ALERTA FUERTE"]
            print(f"{mensajes_alerta[nuevo_nivel_alerta]} - Puntaje: {puntaje_promedio:.1f}%")
        
        #actualizar nivel actual y retornar informacion
        self.nivel_alerta_actual = nuevo_nivel_alerta
        return nuevo_nivel_alerta, puntaje_promedio
    
    def procesar_frame(self, frame):
        #procesa cada fotograma para detectar señales de somnolencia
        alto_frame, ancho_frame = frame.shape[:2]
        #convertir frame a formato rgb requerido por mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #detectar puntos faciales con mediapipe
        results = self.face_mesh.process(rgb_frame)
        
        tiempo_actual = time.time()
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                #calcular ear promedio de ambos ojos
                ear_izq = self.calcular_ear(landmarks, self.PUNTOS_EAR_IZQ)
                ear_der = self.calcular_ear(landmarks, self.PUNTOS_EAR_DER)
                ear_promedio = (ear_izq + ear_der) / 2.0
                
                #actualizar historiales para calculos temporales
                self.historial_ear.append(ear_promedio)
                self.ventana_perclos.append(ear_promedio)
                
                #calcular apertura de boca usando metodo simple
                ratio_boca = self.detectar_apertura_boca(landmarks, alto_frame, ancho_frame)
                self.historial_boca.append(ratio_boca)
                
                #dibujar landmarks de ojos para visualizacion
                for idx in self.PUNTOS_OJO_IZQ:
                    x = int(landmarks[idx].x * ancho_frame)
                    y = int(landmarks[idx].y * alto_frame)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                for idx in self.PUNTOS_OJO_DER:
                    x = int(landmarks[idx].x * ancho_frame)
                    y = int(landmarks[idx].y * alto_frame)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                #dibujar contorno exterior de boca en amarillo
                for idx in self.PUNTOS_BOCA_EXTERIOR:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * ancho_frame)
                        y = int(landmarks[idx].y * alto_frame)
                        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                
                #dibujar contorno interior de boca en cyan
                for idx in self.PUNTOS_BOCA_INTERIOR:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * ancho_frame)
                        y = int(landmarks[idx].y * alto_frame)
                        cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
                
                #dibujar puntos clave de boca en rojo (mas grandes)
                for idx in self.PUNTOS_BOCA_CLAVE:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * ancho_frame)
                        y = int(landmarks[idx].y * alto_frame)
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                
                #dibujar puntos adicionales de boca en magenta
                puntos_adicionales_boca = [i for i in self.TODOS_PUNTOS_BOCA 
                                         if i not in self.PUNTOS_BOCA_EXTERIOR 
                                         and i not in self.PUNTOS_BOCA_INTERIOR 
                                         and i not in self.PUNTOS_BOCA_CLAVE]
                
                for idx in puntos_adicionales_boca:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * ancho_frame)
                        y = int(landmarks[idx].y * alto_frame)
                        cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)
                
                #deteccion del estado de los ojos
                if ear_promedio < self.UMBRAL_OJOS:
                    #iniciar conteo de tiempo con ojos cerrados
                    if self.inicio_ojos_cerrados is None:
                        self.inicio_ojos_cerrados = tiempo_actual
                    self.contador_ojos += 1
                    #mostrar advertencia visual
                    cv2.putText(frame, "OJOS CERRADOS", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    #detectar parpadeo completo si supera umbral de frames
                    if self.contador_ojos >= self.FRAMES_CONSEC_OJOS:
                        self.total_parpadeos += 1
                    #reiniciar contadores de ojos cerrados
                    self.contador_ojos = 0
                    self.inicio_ojos_cerrados = None
                    #mostrar estado normal
                    cv2.putText(frame, "OJOS ABIERTOS", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                #detectar episodios de microsueño
                microsueno, duracion_microsueno = self.detectar_microsueno(tiempo_actual)
                if microsueno:
                    #mostrar advertencia de microsueño con duracion
                    cv2.putText(frame, f"MICROSUEÑO: {duracion_microsueno:.1f}s", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                #deteccion de bostezos
                if ratio_boca > self.UMBRAL_BOCA:
                    #mostrar advertencia de bostezo
                    cv2.putText(frame, "BOSTEZO DETECTADO!", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    #incrementar contador solo una vez por bostezo
                    if not self.bostezo_detectado:
                        self.total_bostezos += 1
                        self.bostezo_detectado = True
                else:
                    #reiniciar bandera al cerrar la boca
                    self.bostezo_detectado = False
                
                #calcular porcentaje de tiempo con ojos cerrados
                perclos = self.calcular_perclos()
                
                #calcular puntaje global de somnolencia combinando todos los factores
                self.puntaje_somnolencia = self.calcular_puntaje_somnolencia(
                    ear_promedio, ratio_boca, perclos, duracion_microsueno if microsueno else 0
                )
                
                #evaluar sistema de alertas segun puntaje
                nivel_alerta, somnolencia_promedio = self.verificar_alertas_somnolencia()
                
                #mostrar puntaje de somnolencia con color segun gravedad
                color_puntaje = (0, 255, 0) if self.puntaje_somnolencia < 40 else (0, 165, 255) if self.puntaje_somnolencia < 70 else (0, 0, 255)
                cv2.putText(frame, f"SOMNOLENCIA: {self.puntaje_somnolencia:.1f}%", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_puntaje, 2)
                
                #mostrar indicador visual de alerta activa
                if nivel_alerta > 0:
                    colores_alerta = [(0, 255, 255), (0, 165, 255), (0, 0, 255)]  #amarillo, naranja, rojo
                    textos_alerta = ["PRECAUCIÓN", "ALERTA", "PELIGRO"]
                    
                    #rectangulo de fondo para texto de alerta
                    cv2.rectangle(frame, (10, 170), (350, 200), colores_alerta[nivel_alerta-1], -1)
                    cv2.putText(frame, f"🚨 {textos_alerta[nivel_alerta-1]} - NIVEL {nivel_alerta}", 
                               (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                #indicador de reproduccion de audio activa
                if self.reproduciendo_alerta:
                    cv2.putText(frame, "♪ REPRODUCIENDO ALERTA ♪", (10, 220),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                #mostrar metricas principales
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
                
                #leyenda de colores para los landmarks de boca
                cv2.putText(frame, "Boca - Amarillo: Exterior, Cyan: Interior", (10, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, "Rojo: Puntos clave, Magenta: Adicionales", (10, 320),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            #mostrar mensaje cuando no se detecta rostro
            cv2.putText(frame, "NO SE DETECTA ROSTRO", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        #devolver frame con anotaciones
        return frame
    
    def iniciar_deteccion(self):
        #funcion principal que ejecuta todo el sistema de deteccion
        #abrir camara web
        cap = cv2.VideoCapture(0)
        
        #verificar si la camara se abrio correctamente
        if not cap.isOpened():
            print("Error: No se pudo acceder a la cámara")
            return
        
        #bucle principal de procesamiento de video
        while True:
            #capturar frame de la camara
            ret, frame = cap.read()
            if not ret:
                print("Error al leer el frame")
                break
            
            #voltear imagen horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            #procesar frame actual con deteccion de somnolencia
            frame = self.procesar_frame(frame)
            
            #mostrar instrucciones de uso en pantalla
            cv2.putText(frame, "Presiona 'q' para salir, 'r' para reiniciar, 's' para silenciar", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            #mostrar frame procesado en ventana
            cv2.imshow('DriverAlert - Detector de Somnolencia', frame)
            
            #controles de teclado para interactuar con la aplicacion
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                #salir del programa
                break
            elif key == ord('r'):
                #reiniciar todos los contadores y estados
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
                #silenciar cualquier alerta activa
                self.detener_alerta()
                print("Alerta detenida")
            elif key == ord('t'):
                #probar sistema de alerta con nivel 2
                self.reproducir_alerta_gradual(2)
                print("Reproduciendo alerta de prueba")
        
        #limpieza de recursos al finalizar
        self.detener_alerta()
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        
        #liberar camara y cerrar ventanas
        cap.release()
        cv2.destroyAllWindows()

#funcion principal del programa
def main():
    #informacion inicial sobre el sistema
    print("DriverAlert - Sistema de detección de somnolencia")
    #crear instancia del detector y ejecutar
    detector = DriverAlert()
    detector.iniciar_deteccion()

if __name__ == "__main__":
    main()