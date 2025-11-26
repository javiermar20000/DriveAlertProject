# DriverAlertProject

Detector de somnolencia en tiempo real para conductores que combina visión por computadora, análisis temporal y alertas sonoras escalonadas. El sistema monitoriza ojos y boca con **MediaPipe Face Mesh**, calcula métricas como EAR/MAR/PERCLOS, identifica microsueños y activa alarmas con distintos niveles de severidad cuando detecta fatiga.

## Características principales
- **Seguimiento facial robusto** usando más de 100 landmarks por rostro; se resaltan ojos y boca para explicar cada medición al usuario.
- **Métricas fisiológicas en paralelo**: EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), PERCLOS, conteo de parpadeos/bostezos y duración de micro‑sueños.
- **Motor de alertas inteligente** con tres niveles (precaución, alerta y peligro), control de enfriamiento y reproducción gradual con `pygame` y el audio `alerta.mp3` (o un tono sintético si no existe).
- **Historiales temporales con `deque`** para suavizar lecturas (ventanas de 3–60 s) y un puntaje de somnolencia combinado (0‑100) que gobierna las alarmas.
- **Dos modos de captura**:
  1. `main.py` usa la webcam local.
  2. `ipwebcam.py` recibe vídeo desde la app IP Webcam de un teléfono Android.
- **Overlay informativo** en OpenCV con métricas, leyendas de colores, contador de eventos y atajos de teclado (`q`, `r`, `s`, `t`).
<img width="644" height="575" alt="imagen" src="https://github.com/user-attachments/assets/ade4fd51-fff5-4df1-8843-f09ca6f73d71" />
<img width="640" height="573" alt="imagen" src="https://github.com/user-attachments/assets/ce41cbbb-1d9e-4dc6-9ad3-310ae9271294" />




## Arquitectura rápida
| Componente | Descripción |
| --- | --- |
| `DriverAlert` (`main.py`) | Núcleo del sistema. Procesa cada frame, calcula métricas, mantiene historiales y gestiona las alertas escalonadas. |
| `DriverAlertTelefono` (`ipwebcam.py`) | Variante que abre un stream MJPEG sobre HTTP (`http://IP:PUERTO/video`). Incluye un pre‑chequeo de conectividad y recordatorios para configurar IP Webcam. |
| `alerta.mp3` | Audio personalizado para las alarmas. Si falta, el código genera un tono sintético multi‑frecuencia. |
| `eyes_model/`, `yawn_model/`, `models/` | Carpeta para modelos entrenados (por ejemplo, experimentos previos con CNNs). El detector actual usa MediaPipe, pero se mantienen para futuras versiones. |

## Requisitos
- Python 3.10+ recomendado.
- Webcam USB/interna **o** teléfono Android con la app [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) conectado a la misma red.
- Dependencias listadas en `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```
  (Incluye `opencv-python`, `numpy`, `mediapipe`, `scipy` y `pygame`).

## Instalación
```bash
git clone <este repositorio>
cd DriveAlertProject
python -m venv .venv
source .venv/bin/activate    # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Coloca `alerta.mp3` (u otro sonido .mp3) en la raíz del proyecto para personalizar las alertas.

## Ejecución
### 1. Webcam local
```bash
python main.py
```
- Se abre la cámara 0; si falla se muestra un mensaje de error.
- Controles en la ventana:
  - `q`: salir.
  - `r`: reiniciar contadores y puntaje.
  - `s`: silenciar cualquier alarma en curso.
  - `t`: reproducir una alerta de prueba (nivel 2).

### 2. Cámara del teléfono
```bash
python ipwebcam.py
```
1. Instala y abre **IP Webcam** en Android, pulsa *Start server* y anota la IP/puerto que se muestran (ej. `192.168.1.124:8080`).
2. Introduce esos datos cuando el script los solicite. El programa probará la conexión antes de comenzar el monitoreo.
3. Los controles dentro de la ventana son idénticos a los del modo webcam.

## Qué muestra la interfaz
- Texto dinámico “OJOS ABIERTOS/CERRADOS”, advertencia de “MICROSUEÑO” con duración y “BOSTEZO DETECTADO”.
- Puntaje de somnolencia coloreado (verde/ámbar/rojo) y barra de alerta con el nivel activo.
- Indicador cuando se reproduce audio.
- Métricas en vivo (`EAR`, `MAR`, `PERCLOS`, parpadeos, bostezos) y leyenda para interpretar los colores de los landmarks.

## Personalización y calibración
- **Umbrales**: ajusta `UMBRAL_OJOS`, `UMBRAL_BOCA` y los valores de `NIVEL_ALERTA_*` en `main.py`/`ipwebcam.py` si necesitas más sensibilidad (por ejemplo, gafas oscuras o iluminación baja).
- **Audio**: cambia `alerta.mp3` por cualquier archivo de tu preferencia. El sistema aplica *fade‑in/fade‑out* y repeticiones según el nivel.
- **Cámara**: modifica el índice de `cv2.VideoCapture()` o el `ip_telefono`/`puerto` por defecto para adaptarlo a tu entorno.

## Referencias de datos
El entrenamiento y validaciones originales utilizaron los siguientes conjuntos públicos:
- [MRL Eye Dataset](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset) – detección de ojos abiertos/cerrados.
- [Yawn Dataset](https://www.kaggle.com/datasets/davidvazquezcic/yawn-dataset) – detección de bostezos.

## Estructura del repositorio
```
DriveAlertProject/
├── main.py              # Detector principal con webcam
├── ipwebcam.py          # Detector vía teléfono/IP Webcam
├── alerta.mp3           # Audio de alerta (opcional, se puede reemplazar)
├── eyes_model/          # Modelos o recursos relacionados con ojos
├── yawn_model/          # Modelos o recursos para bostezos
├── models/              # Otros modelos auxiliares
├── requirements.txt     # Dependencias mínimas
└── README.md
```

## Resolución de problemas
- **“No se pudo acceder a la cámara”**: verifica que ningún otro proceso use la webcam y revisa los permisos del sistema operativo.
- **IP Webcam sin conexión**: confirma que tanto PC como teléfono estén en la misma red WiFi y que el cortafuegos permita el puerto configurado.
- **Audio no suena**: instala los paquetes ALSA/PulseAudio necesarios en Linux, o usa `pygame.mixer.init()` con otros parámetros según tu hardware.

---
Este proyecto no sustituye prácticas de seguridad vial; úsalo como asistente de alerta temprana y mantén siempre la atención en el camino. ¡Maneja con cuidado!
