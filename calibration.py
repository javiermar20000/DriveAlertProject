import time
import numpy as np

def calibrate_user(cap, predict_eye, predict_yawn):
    def capture_state(prompt, seconds=3, samples=15):
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
        preds = sorted(preds)
        trimmed = preds[2:-2]
        return np.mean(trimmed)
    print("\n=== Calibración de usuario ===")
    open_eye_val = capture_state("MANTÉN LOS OJOS ABIERTOS")
    closed_eye_val = capture_state("CIERRA LOS OJOS COMPLETAMENTE")
    yawn_val = capture_state("BOSTEZA (o simula un bostezo fuerte)")
    no_yawn_val = capture_state("RELÁJATE SIN BOSTEZAR (posición neutral)")
    eye_threshold = (closed_eye_val + open_eye_val) / 2
    yawn_threshold = (no_yawn_val + yawn_val) / 2
    print("\n--- Calibración completada ---")
    print(f"Umbral ojos: {eye_threshold:.4f}, Umbral bostezo: {yawn_threshold:.4f}")
    return {
        "eye_threshold": eye_threshold,
        "yawn_threshold": yawn_threshold
    }