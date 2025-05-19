import random
import shutil
import os

def split_dataset(base_dir, output_dir='dataset_etiquetado', train_ratio=0.8):
    """
    Divide las imágenes de cada clase en base_dir en 80% entrenamiento y 20% validación.
    Espera que base_dir tenga subcarpetas (por ejemplo, 'yawn', 'no_yawn').
    Crea la estructura:
        output_dir/
            train/
                yawn/
                no_yawn/
            validation/
                yawn/
                no_yawn/
    """
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    output_dir = os.path.join(os.path.dirname(base_dir), output_dir)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')

    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        images = [f for f in os.listdir(cls_dir) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Crear carpetas destino
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        # Copiar imágenes de entrenamiento
        for img in train_imgs:
            src = os.path.join(cls_dir, img)
            dst = os.path.join(train_dir, cls, img)
            shutil.copy2(src, dst)

        # Copiar imágenes de validación
        for img in val_imgs:
            src = os.path.join(cls_dir, img)
            dst = os.path.join(val_dir, cls, img)
            shutil.copy2(src, dst)

    print(f"Dataset dividido y copiado en: {output_dir}")
    print(f"Entrenamiento: {train_dir}")
    print(f"Validación: {val_dir}")


def label_images(images_dir):
    """Herramienta para etiquetar manualmente imágenes como bostezo o no bostezo"""
    # Verificar si el directorio existe
    if not os.path.exists(images_dir):
        print(f"Error: El directorio {images_dir} no existe.")
        return
    
    # Crear estructura de directorios para las imágenes etiquetadas
    base_dir = os.path.join(os.path.dirname(images_dir), 'dataset_etiquetado')
    
    # Directorios de entrenamiento
    train_dir = os.path.join(base_dir, 'train')
    train_yawn_dir = os.path.join(train_dir, 'yawn')
    train_no_yawn_dir = os.path.join(train_dir, 'no_yawn')
    
    # Directorios de validación
    validation_dir = os.path.join(base_dir, 'validation')
    validation_yawn_dir = os.path.join(validation_dir, 'yawn')
    validation_no_yawn_dir = os.path.join(validation_dir, 'no_yawn')
    
    # Crear los directorios si no existen
    for directory in [train_yawn_dir, train_no_yawn_dir, validation_yawn_dir, validation_no_yawn_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Obtener la lista de archivos de imagen
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"No se encontraron imágenes en {images_dir}")
        return
    
    print(f"Se encontraron {len(image_files)} imágenes para etiquetar.")
    print("Instrucciones:")
    print("- Presiona 'y' si la imagen muestra un bostezo")
    print("- Presiona 'n' si la imagen NO muestra un bostezo")
    print("- Presiona 'v' para marcar como imagen de validación (20% recomendado)")
    print("- Presiona 's' para saltar la imagen actual")
    print("- Presiona 'q' para terminar el etiquetado")
    
    input("Presiona Enter para comenzar el etiquetado...")
    
    # Estadísticas para mostrar al final
    stats = {
        'train_yawn': 0,
        'train_no_yawn': 0,
        'validation_yawn': 0,
        'validation_no_yawn': 0,
        'skipped': 0
    }
    
    # Proceso de etiquetado
    for i, image_file in enumerate(image_files):
        img_path = os.path.join(images_dir, image_file)
        
        # Leer y mostrar la imagen
        img = cv2.imread(img_path)
        if img is None:
            print(f"No se pudo leer la imagen: {img_path}")
            continue
        
        # Redimensionar para visualización si es muy grande
        height, width = img.shape[:2]
        max_display_size = 800
        scale = min(1.0, max_display_size / max(height, width))
        if scale < 1.0:
            img_display = cv2.resize(img, (int(width * scale), int(height * scale)))
        else:
            img_display = img.copy()
        
        # Mostrar imagen y progreso
        window_name = f"Etiquetado ({i+1}/{len(image_files)})"
        cv2.imshow(window_name, img_display)
        
        # Esperar la tecla del usuario
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            # Procesar la tecla presionada
            if key == ord('y') or key == ord('Y'):  # Bostezo
                is_validation = False
                is_yawn = True
                break
            elif key == ord('n') or key == ord('N'):  # No bostezo
                is_validation = False
                is_yawn = False
                break
            elif key == ord('v') or key == ord('V'):  # Imagen de validación
                is_validation = True
                # Esperar segunda tecla para saber si es bostezo o no
                cv2.putText(img_display, "VALIDACION: ¿Bostezo (y) o No bostezo (n)?", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(window_name, img_display)
                while True:
                    subkey = cv2.waitKey(0) & 0xFF
                    if subkey == ord('y') or subkey == ord('Y'):
                        is_yawn = True
                        break
                    elif subkey == ord('n') or subkey == ord('N'):
                        is_yawn = False
                        break
                break
            elif key == ord('s') or key == ord('S'):  # Saltar imagen
                stats['skipped'] += 1
                break
            elif key == ord('q') or key == ord('Q'):  # Salir
                cv2.destroyAllWindows()
                print("Etiquetado terminado por el usuario.")
                print(f"Estadísticas:")
                print(f"- Entrenamiento - Bostezos: {stats['train_yawn']}")
                print(f"- Entrenamiento - No Bostezos: {stats['train_no_yawn']}")
                print(f"- Validación - Bostezos: {stats['validation_yawn']}")
                print(f"- Validación - No Bostezos: {stats['validation_no_yawn']}")
                print(f"- Omitidas: {stats['skipped']}")
                return
        
        # Si decidimos saltar la imagen
        if key == ord('s') or key == ord('S'):
            continue
        
        # Copiar imagen al directorio correspondiente
        file_extension = os.path.splitext(image_file)[1]
        new_filename = f"{os.path.splitext(image_file)[0]}{file_extension}"
        
        if is_validation:
            if is_yawn:
                dest_dir = validation_yawn_dir
                stats['validation_yawn'] += 1
            else:
                dest_dir = validation_no_yawn_dir
                stats['validation_no_yawn'] += 1
        else:
            if is_yawn:
                dest_dir = train_yawn_dir
                stats['train_yawn'] += 1
            else:
                dest_dir = train_no_yawn_dir
                stats['train_no_yawn'] += 1
        
        # Asegurarse de que no haya nombres duplicados
        dest_path = os.path.join(dest_dir, new_filename)
        if os.path.exists(dest_path):
            base_name = os.path.splitext(new_filename)[0]
            new_filename = f"{base_name}_{int(time.time())}{file_extension}"
            dest_path = os.path.join(dest_dir, new_filename)
        
        # Copiar la imagen
        shutil.copy2(img_path, dest_path)
    
    cv2.destroyAllWindows()
    
    print("Etiquetado completado.")
    print(f"Estadísticas:")
    print(f"- Entrenamiento - Bostezos: {stats['train_yawn']}")
    print(f"- Entrenamiento - No Bostezos: {stats['train_no_yawn']}")
    print(f"- Validación - Bostezos: {stats['validation_yawn']}")
    print(f"- Validación - No Bostezos: {stats['validation_no_yawn']}")
    print(f"- Omitidas: {stats['skipped']}")
    
    print(f"\nDatos organizados en: {base_dir}")
    print(f"Para entrenar el modelo, usa los siguientes directorios:")
    print(f"- Directorio de entrenamiento: {train_dir}")
    print(f"- Directorio de validación: {validation_dir}")

def organize_from_videos(video_dir):
    """Extrae frames de videos y prepara para etiquetado"""
    if not os.path.exists(video_dir):
        print(f"Error: El directorio {video_dir} no existe.")
        return
    
    # Crear directorio para los frames extraídos
    frames_dir = os.path.join(os.path.dirname(video_dir), 'frames_extraidos')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Buscar archivos de video
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in os.listdir(video_dir) if os.path.splitext(f)[1].lower() in video_extensions]
    
    if not video_files:
        print(f"No se encontraron videos en {video_dir}")
        return
    
    print(f"Se encontraron {len(video_files)} videos.")
    
    frame_count = 0
    
    # Procesar cada video
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        base_name = os.path.splitext(video_file)[0]
        
        # Abrir el video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"No se pudo abrir el video: {video_path}")
            continue
        
        # Obtener FPS y total de frames
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Procesando {video_file} ({total_frames} frames, {fps} FPS)")
        
        # Extraer frames a intervalos regulares
        frame_interval = max(1, int(fps / 2))  # Extraer 2 frames por segundo
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extraer frame según el intervalo
            if count % frame_interval == 0:
                frame_path = os.path.join(frames_dir, f"{base_name}_frame_{frame_count:05d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_count += 1
            
            count += 1
            
            # Mostrar progreso
            if count % 100 == 0:
                print(f"Procesados {count}/{total_frames} frames...")
        
        cap.release()
    
    print(f"Extracción completada. Se extrajeron {frame_count} frames en total.")
    print(f"Los frames están en: {frames_dir}")
    
    # Preguntar si desea etiquetar los frames ahora
    choice = input("¿Deseas etiquetar los frames ahora? (s/n): ")
    if choice.lower() == 's':
        label_images(frames_dir)
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time
import shutil
from pathlib import Path

class YawnDetector:
    def __init__(self):
        # Ruta donde se guardará el modelo entrenado
        self.model_path = "modelo_detector_bostezos.h5"
        # Tamaño de las imágenes de entrada para el modelo
        self.img_size = (64, 64)
        # Inicializar el detector facial de OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Variable para almacenar el modelo
        self.model = None
        
    def build_model(self):
        """Construye la arquitectura del modelo CNN para la detección de bostezos"""
        model = Sequential([
            # Primera capa convolucional
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 3)),
            MaxPooling2D(2, 2),
            
            # Segunda capa convolucional
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Tercera capa convolucional
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Aplanar los datos para las capas densas
            Flatten(),
            
            # Capas densas
            Dense(512, activation='relu'),
            Dropout(0.5),  # Dropout para reducir el sobreajuste
            Dense(1, activation='sigmoid')  # Salida binaria: bostezo o no bostezo
        ])
        
        # Compilar el modelo
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, train_dir, validation_dir, epochs=15, batch_size=32):
        """Entrena el modelo con los datos proporcionados"""
        # Verificar si el modelo ya ha sido construido
        if self.model is None:
            self.build_model()
            
        # Configurar generadores de datos con aumento de datos
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Flujos de datos de entrenamiento y validación
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary'
        )
        
        # Entrenar el modelo
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )
        
        # Guardar el modelo entrenado
        self.model.save(self.model_path)
        print(f"Modelo guardado en {self.model_path}")
        
        return history
    
    def load_trained_model(self):
        """Carga un modelo previamente entrenado"""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print(f"Modelo cargado desde {self.model_path}")
            return True
        else:
            print(f"No se encontró el modelo en {self.model_path}. Primero debes entrenar el modelo.")
            return False
    
    def preprocess_face(self, face_img):
        """Preprocesa la imagen facial para la entrada del modelo"""
        # Redimensionar la imagen al tamaño esperado por el modelo
        resized = cv2.resize(face_img, self.img_size)
        # Normalizar los valores de píxeles
        normalized = resized / 255.0
        # Expandir dimensiones para coincidir con la entrada esperada del modelo
        return np.expand_dims(normalized, axis=0)
    
    def detect_yawn_in_frame(self, frame):
        """Detecta bostezos en un frame de video"""
        # Convertir a escala de grises para la detección facial
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Procesar cada rostro detectado
        for (x, y, w, h) in faces:
            # Extraer la región del rostro
            face_roi = frame[y:y+h, x:x+w]
            
            # Verificar que la región no esté vacía
            if face_roi.size == 0:
                continue
                
            # Preprocesar la cara para el modelo
            processed_face = self.preprocess_face(face_roi)
            
            # Predecir si hay un bostezo
            if self.model is not None:
                prediction = self.model.predict(processed_face)[0][0]
                # Ajusta el umbral aquí
                if prediction > 0.4:  # Cambia 0.4 por el valor que mejor funcione
                    # Bostezo detectado
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, f'BOSTEZO: {prediction:.2f}', (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    # No hay bostezo
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f'NO BOSTEZO: {1-prediction:.2f}', (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # Si no hay modelo, simplemente marcar la cara
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        return frame
    
    def run_webcam_detection(self):
        """Ejecuta la detección de bostezos usando la webcam en tiempo real"""
        # Verificar si el modelo está cargado
        if self.model is None:
            if not self.load_trained_model():
                print("No se puede iniciar la detección sin un modelo entrenado.")
                return
        
        # Inicializar la webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            return
        
        print("Presiona 'q' para salir")
        
        while True:
            # Capturar frame por frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: No se pudo capturar frame.")
                break
            
            # Detectar bostezos en el frame
            processed_frame = self.detect_yawn_in_frame(frame)
            
            # Mostrar el resultado
            cv2.imshow('Detector de Bostezos', processed_frame)
            
            # Salir si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Liberar la webcam y cerrar ventanas
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Crear instancia del detector
    detector = YawnDetector()
    
    # Opciones para el usuario
    print("Detector de Bostezos en Tiempo Real")
    print("1. Organizar y dividir imágenes (80/20)")
    print("2. Entrenar nuevo modelo")
    print("3. Ejecutar detección con modelo existente")
    
    choice = input("Selecciona una opción (1, 2 o 3): ")
    
    if choice == '1':
        # Rutas predefinidas
        images_dir = r"C:\Users\kirukato\Documents\VisualStudioCode\Inteligencia_artificial\bostesos_xperiment\D3SVideoFrames"
        split_dataset(images_dir)
        print("¡Listo! Ahora puedes entrenar el modelo usando las carpetas generadas.")
        
    elif choice == '2':
        # Rutas predefinidas para entrenamiento y validación
        base_path = r"C:\Users\kirukato\Documents\VisualStudioCode\Inteligencia_artificial\bostesos_xperiment\dataset_etiquetado"
        train_dir = os.path.join(base_path, "train")
        validation_dir = os.path.join(base_path, "validation")
        
        # Verificar si las rutas existen
        if not os.path.exists(train_dir) or not os.path.exists(validation_dir):
            print("Error: Las rutas especificadas no existen. Ejecuta primero la opción 1.")
        else:
            detector.build_model()
            detector.train_model(train_dir, validation_dir, epochs=30)
            run_detection = input("¿Deseas ejecutar la detección ahora? (s/n): ")
            if run_detection.lower() == 's':
                detector.run_webcam_detection()
    
    elif choice == '3':
        detector.run_webcam_detection()
    
    else:
        print("Opción no válida.")