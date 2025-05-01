import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import cv2
import numpy as np
import os

# --- Cargar o entrenar modelo de ojos ---
if os.path.exists("models/eye_model_trained.h5"):
    eye_classifier = load_model("models/eye_model_trained.h5")
    print("Modelo de ojos cargado desde archivo.")
else:
    print("Entrenando modelo de ojos...")
    eye_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    eye_model.trainable = False

    eye_classifier = Sequential([
        eye_model,
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])

    eye_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

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
    eye_classifier.fit(eye_train_generator, validation_data=eye_val_generator, epochs=5)

    os.makedirs("models", exist_ok=True)
    eye_classifier.save("models/eye_model_trained.h5")
    print("Modelo de ojos entrenado y guardado.")

# --- Cargar o entrenar modelo de bostezo ---
if os.path.exists("models/yawn_model_trained.h5"):
    yawn_model = load_model("models/yawn_model_trained.h5")
    print("Modelo de bostezo cargado desde archivo.")
else:
    print("Entrenando modelo de bostezo...")
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    yawn_model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

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
    yawn_model.fit(train_generator, validation_data=val_generator, epochs=5)

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

# --- Loop de cámara ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (224, 224))
    img = preprocess_input(resized_frame)
    img = np.expand_dims(img, axis=0)

    yawn_pred = yawn_model.predict(img)[0][0]
    eye_pred = eye_classifier.predict(img)[0][0]

    yawn_text = "Bostezo" if yawn_pred > 0.5 else "No bostezo"
    eye_text = "Ojos abiertos" if eye_pred > 0.5 else "Ojos cerrados"

    cv2.putText(frame, f"{yawn_text} ({yawn_pred:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"{eye_text} ({eye_pred:.2f})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Detección de Sueño", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()