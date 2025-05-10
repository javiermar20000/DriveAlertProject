
import tensorflow as tf # Importa TensorFlow para el uso de modelos de deep learning
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Herramienta de Keras para aplicar aumentos de imagen durante el entrenamiento
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input # Carga el modelo MobileNet preentrenado y su función de preprocesamiento
from tensorflow.keras.models import Sequential, load_model # Permite crear modelos secuenciales y cargar modelos guardados
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # Capas comunes para redes convolucionales: totalmente conectada y pooling global
from tensorflow.keras.applications import ResNet50 # También importa otro modelo preentrenado muy usado: ResNet50
import cv2 # Importa OpenCV para captura y procesamiento de imágenes desde la cámara
import numpy as np # Importa NumPy para manejo de arreglos y cálculos numéricos
import os # Permite interactuar con archivos y carpetas del sistema operativo
import tkinter as tk # Importa Tkinter para construir interfaces gráficas simples
from PIL import Image, ImageTk # Pillow: permite cargar y adaptar imágenes para usarlas con Tkinter
import pygame # Pygame: se usa aquí posiblemente para reproducir sonidos (alarmas, etc.)
import threading # Módulo para trabajar con hilos (ejecución paralela)
import time # Módulo para manejar tiempos de espera o medir duración
from sklearn.metrics import classification_report, confusion_matrix # cargamos metricas de evaluacion
import matplotlib.pyplot as plt # cargar graficos
import seaborn as sns # graficos mejorados


eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Cargar modelos existentes ---
# Se verifica si ya existe un modelo previamente entrenado para la detección de ojos

if os.path.exists("models/eye_model_trained.h5"):
    # Si el archivo del modelo existe, se carga desde el disco
    eye_classifier = load_model("models/eye_model_trained.h5")
    print("Modelo de ojos cargado desde archivo.")
else:
    # Si no existe el modelo, se prepara uno nuevo desde MobileNet
    print("Entrenando modelo de ojos...")

    # Se carga la arquitectura MobileNet con pesos preentrenados (sin la parte superior de clasificación)
    eye_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Se congelan los pesos de MobileNet para que no se entrenen
    eye_model.trainable = False

    # Se crea un modelo secuencial añadiendo la base MobileNet, una capa de pooling y una capa final de clasificación binaria
    eye_classifier = Sequential([
        eye_model,                         # Modelo base (MobileNet sin entrenar)
        GlobalAveragePooling2D(),         # Capa para reducir la salida de la CNN a un vector
        Dense(1, activation='sigmoid')    # Capa de salida para clasificar entre 2 clases (ojo abierto vs cerrado)
    ])

    # Se crea un generador de datos de imagen con aumentos para entrenar el modelo de detección de ojos
    eye_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Aplica la función de preprocesamiento de MobileNet (escalado y normalización)
    rotation_range=10,                        # Rota aleatoriamente las imágenes hasta ±10 grados
    width_shift_range=0.1,                    # Desplaza aleatoriamente las imágenes horizontalmente hasta un 10% del ancho
    height_shift_range=0.1,                   # Desplaza aleatoriamente las imágenes verticalmente hasta un 10% del alto
    shear_range=0.2,                          # Aplica una transformación de corte (shear) en las imágenes hasta 0.2 radianes
    zoom_range=0.2,                           # Aplica un zoom aleatorio en el rango de ±20%
    horizontal_flip=True,                     # Invierte las imágenes horizontalmente aleatoriamente
    fill_mode='nearest',                      # Rellena los pixeles vacíos generados por las transformaciones con el valor más cercano
    validation_split=0.2                      # Reserva el 20% de los datos para validación (útil cuando se usan desde carpetas)
    )

    # Crea un generador de datos para entrenamiento a partir del directorio 'eyes_model'
    # Usa las transformaciones definidas en eye_datagen (preprocesamiento + aumentos)
    eye_train_generator = eye_datagen.flow_from_directory(
        'eyes_model',              # Ruta al directorio que contiene subcarpetas con imágenes (una por clase)
        target_size=(224, 224),    # Redimensiona todas las imágenes a 224x224 píxeles
        batch_size=16,             # Cantidad de imágenes por lote (batch) durante el entrenamiento
        class_mode='binary',       # Clasificación binaria (por ejemplo: ojos abiertos vs cerrados)
        subset='training'          # Indica que este generador usará el 80% de los datos para entrenamiento
    )

    # Crea un generador de datos para validación desde el mismo directorio
    eye_val_generator = eye_datagen.flow_from_directory(
        'eyes_model',              # Usa el mismo conjunto de imágenes
        target_size=(224, 224),    # Mismo tamaño de entrada
        batch_size=16,             # Mismo tamaño de batch
        class_mode='binary',       # También es clasificación binaria
        subset='validation'        # Usa el 20% restante de los datos para validación
    )

    # Compila el modelo: usa el optimizador Adam, función de pérdida binaria y métrica de precisión
    eye_classifier.compile(
        optimizer='adam',                # Algoritmo de optimización
        loss='binary_crossentropy',     # Función de pérdida adecuada para clasificación binaria
        metrics=['accuracy']            # Métrica a mostrar durante el entrenamiento y validación
    )

    # Entrena el modelo usando los generadores de entrenamiento y validación
    eye_classifier.fit(
        eye_train_generator,             # Datos de entrenamiento
        validation_data=eye_val_generator,  # Datos de validación
        epochs=30                        # Número de veces que el modelo verá todos los datos (épocas)
    )

    # Crea el directorio "models" si no existe aún
    os.makedirs("models", exist_ok=True)

    # Guarda el modelo entrenado en un archivo HDF5 para uso posterior
    eye_classifier.save("models/eye_model_trained.h5")

    # Imprime mensaje de confirmación
    print("Modelo de ojos entrenado y guardado.")

    # --- Evaluación del modelo de ojos ---

    # Obtener predicciones
    val_generator_reset = eye_val_generator.reset()  # Reinicia el generador
    y_pred_probs = eye_classifier.predict(eye_val_generator)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Convertir a etiquetas 0 o 1

    # Obtener etiquetas verdaderas
    y_true = eye_val_generator.classes

    # Reporte de clasificación
    print("Reporte de Clasificación - Modelo de Ojos:")
    print(classification_report(y_true, y_pred, target_names=list(eye_val_generator.class_indices.keys())))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=eye_val_generator.class_indices.keys(), yticklabels=eye_val_generator.class_indices.keys())
    plt.title("Matriz de Confusión - Modelo de Ojos")
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.show()

# Verifica si ya existe un modelo entrenado de detección de bostezos
if os.path.exists("models/yawn_model_trained.h5"):
    # Si el archivo existe, lo carga directamente
    yawn_model = load_model("models/yawn_model_trained.h5")
    print("Modelo de bostezo cargado desde archivo.")
else:
    # Si no existe, se entrena un nuevo modelo
    print("Entrenando modelo de bostezo...")

    # Carga la arquitectura base de MobileNet sin las capas superiores (top)
    # y con pesos preentrenados en ImageNet
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Congela las capas del modelo base para que no se entrenen
    base_model.trainable = False

    # Construye el modelo secuencial agregando la base, un GlobalAveragePooling2D y una capa de salida densa
    yawn_model = Sequential([
        base_model,                        # Modelo base MobileNet
        GlobalAveragePooling2D(),         # Reduce la dimensionalidad después del modelo base
        Dense(1, activation='sigmoid')    # Capa de salida con activación sigmoide para clasificación binaria
    ])

    # Define el generador de datos con aumentos para mejorar la generalización
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # Preprocesamiento específico de MobileNet
        rotation_range=10,                        # Rotaciones aleatorias hasta ±10 grados
        width_shift_range=0.1,                    # Desplazamiento horizontal aleatorio del 10%
        height_shift_range=0.1,                   # Desplazamiento vertical aleatorio del 10%
        shear_range=0.2,                          # Transformaciones de corte
        zoom_range=0.2,                           # Zoom aleatorio
        horizontal_flip=True,                     # Volteo horizontal aleatorio
        fill_mode='nearest',                      # Cómo rellenar los píxeles vacíos después de las transformaciones
        validation_split=0.2                      # Reserva el 20% de los datos para validación
    )

    # Generador de imágenes para entrenamiento (80% de los datos)
    train_generator = train_datagen.flow_from_directory(
        'yawn_model',               # Carpeta raíz con subdirectorios para cada clase
        target_size=(224, 224),     # Redimensiona las imágenes a 224x224
        batch_size=16,              # Tamaño de cada batch
        class_mode='binary',        # Clasificación binaria
        subset='training'           # Usa el subconjunto de entrenamiento
    )

    # Generador de imágenes para validación (20% de los datos)
    val_generator = train_datagen.flow_from_directory(
        'yawn_model',               # Mismo directorio
        target_size=(224, 224),     # Mismo tamaño de imagen
        batch_size=16,              # Mismo tamaño de batch
        class_mode='binary',        # Clasificación binaria
        subset='validation'         # Subconjunto de validación
    )

    # Compila el modelo con el optimizador Adam y función de pérdida binaria
    yawn_model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )

    # Entrena el modelo durante 30 épocas usando los generadores
    yawn_model.fit(
        train_generator, 
        validation_data=val_generator, 
        epochs=30
    )

    # Crea el directorio "models" si no existe
    os.makedirs("models", exist_ok=True)

    # Guarda el modelo entrenado en un archivo HDF5
    yawn_model.save("models/yawn_model_trained.h5")

    # Muestra un mensaje confirmando que se guardó el modelo
    print("Modelo de bostezo entrenado y guardado.")

    # --- Evaluación del modelo de bostezo ---

    # Reinicia el generador
    val_generator_reset = val_generator.reset()

    # Predicciones
    yawn_pred_probs = yawn_model.predict(val_generator)
    yawn_pred = (yawn_pred_probs > 0.5).astype(int).flatten()

    # Verdaderos
    yawn_true = val_generator.classes

    # Reporte de clasificación
    print("Reporte de Clasificación - Modelo de Bostezo:")
    print(classification_report(yawn_true, yawn_pred, target_names=list(val_generator.class_indices.keys())))

    #   Matriz de confusión
    cm_yawn = confusion_matrix(yawn_true, yawn_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_yawn, annot=True, fmt='d', cmap='Purples', xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
    plt.title("Matriz de Confusión - Modelo de Bostezo")
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.show()

# --- Funciones de predicción ---

# Función para predecir si hay un bostezo en una imagen
def predict_yawn(image):
    # Redimensiona la imagen a 224x224 píxeles (requisito del modelo MobileNet)
    img = cv2.resize(image, (224, 224))
    # Aplica el preprocesamiento específico de MobileNet (normalización, etc.)
    img = preprocess_input(img)
    # Expande la dimensión para que tenga la forma (1, 224, 224, 3), necesaria para el modelo
    img = np.expand_dims(img, axis=0)
    # Realiza la predicción con el modelo de bostezo y devuelve el valor de salida (probabilidad)
    return yawn_model.predict(img)[0][0]

# Función para predecir si el ojo está abierto o cerrado en una imagen
def predict_eye(image):
    # Redimensiona la imagen a 224x224 píxeles
    img = cv2.resize(image, (224, 224))
    # Preprocesa la imagen como requiere MobileNet
    img = preprocess_input(img)
    # Agrega una dimensión adicional para representar el batch (1 imagen)
    img = np.expand_dims(img, axis=0)
    # Predice con el modelo de ojos y devuelve la probabilidad de que esté abierto/cerrado
    return eye_classifier.predict(img)[0][0]

# --- Calibración inicial del usuario ---
def calibrate_user(cap):  # Función que realiza la calibración inicial usando la cámara capturada en `cap`
    
    # Función interna para capturar un estado del usuario (ojos abiertos, cerrados, etc.)
    def capture_state(prompt, seconds=3, samples=15):
        # Muestra una instrucción al usuario
        print(f"\n[Instrucción] {prompt}")
        print("Capturando imágenes en...")
        
        # Cuenta regresiva antes de comenzar la captura
        for i in range(seconds, 0, -1):
            print(f"{i}...")
            time.sleep(1)

        preds = []  # Lista para guardar las predicciones

        # Captura múltiples imágenes y realiza predicciones
        for _ in range(samples):
            ret, frame = cap.read()  # Lee un frame de la cámara
            if ret:  # Si se leyó correctamente
                if "ojos" in prompt.lower():  # Si el prompt es sobre los ojos
                    preds.append(predict_eye(frame))  # Usa el modelo de ojos
                elif "bostezo" in prompt.lower():  # Si el prompt es sobre bostezo
                    preds.append(predict_yawn(frame))  # Usa el modelo de bostezo
            time.sleep(0.3)  # Espera entre capturas para no repetir frames

        # Ordena las predicciones para eliminar valores extremos
        preds = sorted(preds)
        trimmed = preds[2:-2]  # Elimina los 2 valores más altos y los 2 más bajos
        return np.mean(trimmed)  # Devuelve el promedio como valor representativo

    print("\n=== Calibración de usuario ===")

    # Captura el valor promedio con ojos abiertos
    open_eye_val = capture_state("MANTÉN LOS OJOS ABIERTOS")
    # Captura el valor promedio con ojos cerrados
    closed_eye_val = capture_state("CIERRA LOS OJOS COMPLETAMENTE")
    # Captura el valor promedio al bostezar
    yawn_val = capture_state("BOSTEZA (o simula un bostezo fuerte)")
    # Captura el valor promedio con la boca relajada (sin bostezo)
    no_yawn_val = capture_state("RELÁJATE SIN BOSTEZAR (posición neutral)")

    # Calcula el umbral para los ojos como el promedio entre ojos abiertos y cerrados
    eye_threshold = (closed_eye_val + open_eye_val) / 2
    # Calcula el umbral para bostezo como el promedio entre neutral y bostezo
    yawn_threshold = (no_yawn_val + yawn_val) / 2

    # Muestra los resultados de la calibración
    print("\n--- Calibración completada ---")
    print(f"Umbral ojos: {eye_threshold:.4f}, Umbral bostezo: {yawn_threshold:.4f}")

    # Devuelve los umbrales como diccionario
    return {
        "eye_threshold": eye_threshold,
        "yawn_threshold": yawn_threshold
    }

# --- Interfaz Tkinter con calibración integrada ---
class SleepDetectorApp:  # Define una clase llamada SleepDetectorApp, que representa la aplicación de detección de sueño.
    def __init__(self, root):  # Método constructor que recibe la ventana principal de Tkinter como argumento.
        self.root = root  # Guarda la referencia a la ventana principal.
        self.root.title("Detección de Sueño")  # Establece el título de la ventana.

        self.eyes_closed_start_time = None  # Inicializa la variable para registrar el momento en que se cierran los ojos.
        self.eyes_closed_duration_threshold = 5  # Umbral de duración (en segundos) para considerar que una persona se ha dormido.

        self.video_label = tk.Label(root)  # Crea una etiqueta para mostrar el video capturado.
        self.video_label.pack()  # Añade la etiqueta de video a la ventana.

        self.status_label = tk.Label(root, text="", font=("Helvetica", 14))  # Crea una etiqueta para mostrar el estado (como "Durmiendo").
        self.status_label.pack(pady=10)  # Añade la etiqueta de estado con un margen vertical.

        self.close_button = tk.Button(root, text="Cerrar", command=self.close_app)  # Crea un botón para cerrar la aplicación.
        self.close_button.pack()  # Añade el botón a la ventana.

        self.cap = cv2.VideoCapture(0)  # Inicia la captura de video desde la cámara predeterminada (índice 0).

        pygame.mixer.init()  # Inicializa el mezclador de audio de Pygame.
        pygame.mixer.music.load("alerta.mp3")  # Carga un archivo de sonido que se usará como alerta.

        self.thresholds = calibrate_user(self.cap)  # Llama a una función para calibrar el sistema con el usuario, usando la cámara.

        # Cargar el clasificador Haar para los ojos
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")  
        # Carga un clasificador preentrenado para detectar ojos usando el algoritmo Haar Cascade.

        # Cargar el clasificador Haar para la detección de rostros
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
        # Carga un clasificador preentrenado para detectar rostros.

        self.update_video()  # Llama a un método que actualiza continuamente el video y procesa los fotogramas.

    def play_alert_sound(self):  # Define un método para reproducir el sonido de alerta.
        if not pygame.mixer.music.get_busy():  # Verifica si no se está reproduciendo ya un sonido.
            pygame.mixer.music.play()  # Reproduce el archivo de sonido cargado previamente.

    def update_video(self): # Método principal que se ejecuta continuamente para analizar el video.
        ret, frame = self.cap.read()  # Lee un frame (fotograma) de la cámara.
        if not ret:  # Si no se pudo capturar el frame...
            return  # ...termina la función.

        img_resized = cv2.resize(frame, (224, 224))  # Redimensiona el frame a 224x224 píxeles (tamaño esperado por el modelo).
        yawn_pred = predict_yawn(img_resized)  # Predice si hay bostezo en el frame.
        eye_pred = predict_eye(img_resized)  # Predice si los ojos están abiertos o cerrados.

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convierte el frame a escala de grises.
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)  # Detecta rostros en la imagen.

        if len(faces) > 0:  # Si se detecta al menos un rostro...
            yawn_text = "Bostezo" if yawn_pred > self.thresholds["yawn_threshold"] else "No bostezo"  # Define el texto según el umbral de bostezo.
            eye_text = "Ojos abiertos" if eye_pred > self.thresholds["eye_threshold"] else "Ojos cerrados"  # Define el texto según el umbral de ojos.

            if yawn_pred > self.thresholds["yawn_threshold"]:  # Si se detecta un bostezo...
                threading.Thread(target=self.play_alert_sound, daemon=True).start()  # Reproduce alerta en un hilo separado.

            if eye_pred < self.thresholds["eye_threshold"]: # Si los ojos están cerrados...
                if self.eyes_closed_start_time is None: # Si no hay tiempo registrado aún...
                    self.eyes_closed_start_time = time.time() # Registra el tiempo actual.
                elif time.time() - self.eyes_closed_start_time >= self.eyes_closed_duration_threshold: # Si han pasado más de 5 segundos
                    threading.Thread(target=self.play_alert_sound, daemon=True).start() # Reproduce alerta.
            else:
                self.eyes_closed_start_time = None # Si los ojos están abiertos, reinicia el temporizador.
        else:
            yawn_text = "" # No hay rostro → vacío.
            eye_text = "" # No hay rostro → vacío.
            self.eyes_closed_start_time = None  # Reinicia el temporizador si no hay rostro.

        # === Detección de ojos con rectángulo verde ===
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convierte el frame a escala de grises nuevamente
        eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5) # Detecta ojos en el frame.

        for (ex, ey, ew, eh) in eyes: # Dibuja rectángulo y texto por cada ojo detectado.
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2) # Rectángulo verde alrededor del ojo.
            cv2.putText(frame, "Ojos detectados", (ex, ey + eh + 20),  # Texto debajo del ojo.
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # === Detección de rostros con rectángulo rojo ===
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5) # Detecta nuevamente rostros.

        for (fx, fy, fw, fh) in faces: # Dibuja rectángulo y texto por cada rostro detectado.
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2) # Rectángulo rojo alrededor del rostro.
            cv2.putText(frame, "Rostro detectado", (fx, fy + fh + 20),  # Texto debajo del rostro.
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        # Mostrar texto en el frame solo si hay rostros
        if len(faces) == 0: # Si no hay rostros detectados...
            cv2.putText(frame, "Rostro no detectado", (10, 30), # Muestra mensaje de error en pantalla
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"{yawn_text} ({yawn_pred:.2f})", (10, 30), # Muestra estado de bostezo con probabilidad.
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"{eye_text} ({eye_pred:.2f})", (10, 60), # Muestra estado de ojos con probabilidad
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if len(faces) == 0:
            self.status_label.config(text="Rostro no detectado") # Actualiza etiqueta de estado en la interfaz.
        else:
            self.status_label.config(text=f"{yawn_text} | {eye_text}") # Actualiza la etiqueta con el estado de ojos y bostezo.

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convierte el frame de BGR a RGB para mostrarlo correctamente.
        img = Image.fromarray(frame) # Convierte el frame en un objeto de imagen de PIL.
        imgtk = ImageTk.PhotoImage(image=img) # Convierte la imagen PIL en formato compatible con Tkinter.

        self.video_label.imgtk = imgtk # Guarda la referencia de la imagen en el widget para evitar que se elimine.
        self.video_label.configure(image=imgtk) # Muestra la imagen en el widget de video.

        self.root.after(10, self.update_video) # Llama de nuevo a este método después de 10 ms para actualizar el video.

    def close_app(self): # Define el método `close_app` para cerrar la aplicación correctamente.
        self.cap.release() # Libera el recurso de la cámara (VideoCapture) para evitar bloqueos.
        self.root.destroy() # Cierra la ventana principal de la interfaz gráfica Tkinter.

# --- Ejecutar interfaz ---
if __name__ == "__main__": # Verifica si el script se está ejecutando directamente (no importado).
    root = tk.Tk() # Crea la ventana principal de la aplicación usando Tkinter.
    app = SleepDetectorApp(root) # Crea una instancia de la clase `SleepDetectorApp` pasando la ventana.
    root.mainloop() # Inicia el bucle principal de la interfaz gráfica, manteniéndola activa.


