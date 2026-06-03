"""
Práctica 4 — Detección en Tiempo Real
======================================
Uso:
    python deteccion_tiempo_real.py [opciones]

Opciones:
    --modelo    yolo | faster | ambos        (default: yolo)
    --camara    índice de cámara             (default: 0)
    --imgsz     tamaño de imagen en px       (default: 480)
    --conf      umbral de confianza          (default: 0.4)
    --grabar    graba demo_video.mp4 y cierra al terminar
    --segundos  duración de grabación        (default: 30)

Teclas durante la ejecución:
    Q / ESC   → salir
    S         → guardar snapshot
    M         → cambiar modelo (yolo → faster → ambos)
    +/-       → subir/bajar umbral de confianza

Ejemplos:
    python deteccion_tiempo_real.py
    python deteccion_tiempo_real.py --modelo faster
    python deteccion_tiempo_real.py --grabar --segundos 30
    python deteccion_tiempo_real.py --camara 1 --imgsz 320
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torchvision
import torchvision.models.detection as detection
import torchvision.transforms.functional as TF
from PIL import Image
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────
FASTER_CHECKPOINT = "checkpoints/faster_rcnn/best_model.pth"
YOLO_CHECKPOINT   = "checkpoints/yolo/yolov8n_run/weights/best.pt"

CLASS_NAMES_FASTER = {0: "background", 1: "person", 2: "chair", 3: "laptop"}
CLASS_NAMES_YOLO   = ["person", "chair", "laptop"]

# IDs en modelo COCO base (sin fine-tuning)
COCO_FASTER_MAP = {1: "person", 62: "chair", 73: "laptop"}
COCO_YOLO_MAP   = {0: "person", 56: "chair", 63: "laptop"}
YOLO_CLASSES    = [0, 56, 63]

# Colores BGR por clase
COLORS = {
    "person":  (50,  205,  50),
    "chair":   (255, 191,   0),
    "laptop":  (0,   215, 255),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────
# CARGAR MODELOS
# ──────────────────────────────────────────────────────────────
def load_faster_rcnn():
    print("Cargando Faster R-CNN ResNet50-FPN-V2...", end=" ", flush=True)
    weights = detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model   = detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

    if os.path.exists(FASTER_CHECKPOINT):
        ckpt  = torch.load(FASTER_CHECKPOINT, map_location=DEVICE)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        print(f"checkpoint '{FASTER_CHECKPOINT}' cargado.")
    else:
        print("pesos COCO base.")

    model.roi_heads.score_thresh       = 0.4
    model.roi_heads.nms_thresh         = 0.45
    model.roi_heads.detections_per_img = 50
    model.eval().to(DEVICE)
    return model


def load_yolo():
    print("Cargando YOLOv8n...", end=" ", flush=True)
    path = YOLO_CHECKPOINT if os.path.exists(YOLO_CHECKPOINT) else "yolov8n.pt"
    model = YOLO(path)
    print(f"'{path}' cargado.")
    return model


# ──────────────────────────────────────────────────────────────
# INFERENCIA
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_faster(model, frame_bgr, conf_thresh):
    """Devuelve lista de (x1,y1,x2,y2,name,score)."""
    # Resize para no saturar CPU
    h, w = frame_bgr.shape[:2]
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_t   = TF.to_tensor(Image.fromarray(img_rgb)).to(DEVICE)

    model.roi_heads.score_thresh = conf_thresh
    preds = model([img_t])[0]

    is_finetuned = os.path.exists(FASTER_CHECKPOINT)
    dets = []
    for box, label, score in zip(
        preds["boxes"].cpu().numpy(),
        preds["labels"].cpu().numpy(),
        preds["scores"].cpu().numpy(),
    ):
        label = int(label)
        if is_finetuned:
            name = CLASS_NAMES_FASTER.get(label)
        else:
            name = COCO_FASTER_MAP.get(label)
        if name is None or name == "background":
            continue
        dets.append((*box, name, float(score)))
    return dets


def predict_yolo(model, frame_bgr, conf_thresh, imgsz):
    """Devuelve lista de (x1,y1,x2,y2,name,score)."""
    is_finetuned = os.path.exists(YOLO_CHECKPOINT)
    cls_filter   = None if is_finetuned else YOLO_CLASSES

    results = model.predict(
        frame_bgr,
        conf    = conf_thresh,
        iou     = 0.45,
        imgsz   = imgsz,
        classes = cls_filter,
        verbose = False,
    )[0]

    dets = []
    for box_data in results.boxes:
        x1, y1, x2, y2 = box_data.xyxy[0].cpu().numpy()
        score   = float(box_data.conf[0])
        cls_idx = int(box_data.cls[0])

        if is_finetuned:
            name = CLASS_NAMES_YOLO[cls_idx] if cls_idx < len(CLASS_NAMES_YOLO) else None
        else:
            name = COCO_YOLO_MAP.get(cls_idx)
        if name is None:
            continue
        dets.append((x1, y1, x2, y2, name, float(score)))
    return dets


# ──────────────────────────────────────────────────────────────
# DIBUJO
# ──────────────────────────────────────────────────────────────
def draw_detections(frame, dets, model_label, fps, conf_thresh):
    out = frame.copy()

    for x1, y1, x2, y2, name, score in dets:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = COLORS.get(name, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        txt = f"{name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Barra de estado inferior
    h = out.shape[0]
    cv2.rectangle(out, (0, h - 30), (out.shape[1], h), (20, 20, 40), -1)
    status = (f"[{model_label}]  FPS: {fps:.1f}  "
              f"Dets: {len(dets)}  Conf: {conf_thresh:.2f}  "
              f"[Q=salir  S=foto  M=modelo  +/-=conf]")
    cv2.putText(out, status, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 136), 1)
    return out


# ──────────────────────────────────────────────────────────────
# DETECCIÓN DE BACKEND DE DISPLAY
# ──────────────────────────────────────────────────────────────
def check_display():
    """
    Verifica si cv2.imshow puede abrir ventanas.
    La barra negra ocurre cuando el backend de Qt/GTK está instalado
    pero no tiene acceso al display (ej. SSH sin -X, o conda sin
    libGL). El test abre una ventana de 1x1 y comprueba que no crashea.
    """
    try:
        test = np.zeros((1, 1, 3), dtype=np.uint8)
        cv2.imshow("__test__", test)
        cv2.waitKey(1)
        cv2.destroyWindow("__test__")
        return True
    except cv2.error:
        return False


# ──────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ──────────────────────────────────────────────────────────────
def run(args):
    # Cargar modelos según modo
    faster_model = None
    yolo_model   = None

    modelo_actual = args.modelo

    if modelo_actual in ("faster", "ambos"):
        faster_model = load_faster_rcnn()
    if modelo_actual in ("yolo", "ambos"):
        yolo_model = load_yolo()

    # Abrir cámara
    print(f"\nAbriendo cámara {args.camara}...")
    cap = cv2.VideoCapture(args.camara)

    # Algunas cámaras necesitan un segundo intento con backend explícito
    if not cap.isOpened():
        print(f"  Reintentando con CAP_DSHOW...")
        cap = cv2.VideoCapture(args.camara, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"  Reintentando con CAP_V4L2...")
        cap = cv2.VideoCapture(args.camara, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"ERROR: No se pudo abrir la cámara {args.camara}. "
                 f"Prueba con --camara 1 o --camara 2.")

    # Ajustar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Buffer mínimo para reducir lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Verificar display
    display_ok = check_display()
    if not display_ok:
        print("AVISO: cv2.imshow no puede abrir ventanas en este entorno.")
        print("       Los frames se guardarán como snapshots cada 2 segundos.")
        print("       Para habilitar display: ejecuta desde terminal (no SSH sin -X)")

    # Preparar grabación si se pidió
    writer = None
    if args.grabar:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("demo_video.mp4", fourcc, 20, (640, 480))
        print(f"Grabando {args.segundos}s → demo_video.mp4")

    # Crear ventana ANTES del loop (fix para barra negra)
    WIN_NAME = "Practica 4 - Deteccion en Tiempo Real"
    if display_ok:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 960, 540)

    os.makedirs("snapshots", exist_ok=True)

    # ── Variables de estado ──────────────────────────────────
    conf_thresh  = args.conf
    frame_count  = 0
    fps_list     = []
    last_snap_t  = time.time()
    modelos_ciclo = ["yolo", "faster", "ambos"]

    print(f"\nModelo activo : {modelo_actual.upper()}")
    print(f"Conf threshold: {conf_thresh:.2f}")
    print(f"imgsz         : {args.imgsz}px")
    print(f"Dispositivo   : {DEVICE}")
    print("\nControles: Q/ESC=salir  S=foto  M=cambiar modelo  +/-=confianza\n")

    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame de la cámara.")
            break

        # Resize para inferencia más rápida
        frame_small = cv2.resize(frame, (args.imgsz, int(args.imgsz * 480 / 640)))

        t0 = time.time()

        # ── Inferencia ───────────────────────────────────────
        all_dets = []

        if modelo_actual in ("yolo", "ambos") and yolo_model is not None:
            # Escalar coordenadas de vuelta al frame original
            dets_small = predict_yolo(yolo_model, frame_small, conf_thresh, args.imgsz)
            sx = frame.shape[1] / frame_small.shape[1]
            sy = frame.shape[0] / frame_small.shape[0]
            for x1,y1,x2,y2,name,score in dets_small:
                all_dets.append((x1*sx, y1*sy, x2*sx, y2*sy, name, score))

        if modelo_actual in ("faster", "ambos") and faster_model is not None:
            # Faster R-CNN maneja el resize internamente (FPN)
            all_dets += predict_faster(faster_model, frame_small, conf_thresh)

        # ── FPS ──────────────────────────────────────────────
        elapsed = time.time() - t0
        fps_cur  = 1.0 / elapsed if elapsed > 0 else 0
        fps_list.append(fps_cur)
        fps_avg = float(np.mean(fps_list[-30:]))

        # ── Dibujar ──────────────────────────────────────────
        frame_out = draw_detections(frame, all_dets, modelo_actual.upper(),
                                    fps_avg, conf_thresh)

        # ── Mostrar / guardar ─────────────────────────────────
        if display_ok:
            cv2.imshow(WIN_NAME, frame_out)
        else:
            # Sin display: guardar un snapshot cada 2 segundos
            if time.time() - last_snap_t > 2.0:
                path = f"snapshots/auto_{frame_count:05d}.jpg"
                cv2.imwrite(path, frame_out)
                print(f"\r  Auto-snapshot: {path}  FPS={fps_avg:.1f}  Dets={len(all_dets)}", end="")
                last_snap_t = time.time()

        # ── Grabación ────────────────────────────────────────
        if writer is not None:
            writer.write(cv2.resize(frame_out, (640, 480)))
            elapsed_total = time.time() - t_start
            if elapsed_total >= args.segundos:
                print(f"\nGrabación terminada ({args.segundos}s).")
                break

        # ── Teclas ───────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF if display_ok else 0xFF

        if key in (ord("q"), 27):       # Q o ESC → salir
            print("\nCerrando...")
            break

        elif key == ord("s"):           # S → snapshot manual
            path = f"snapshots/snap_{frame_count:05d}.jpg"
            cv2.imwrite(path, frame_out)
            print(f"\nSnapshot guardado: {path}")

        elif key == ord("m"):           # M → cambiar modelo
            idx = modelos_ciclo.index(modelo_actual)
            modelo_actual = modelos_ciclo[(idx + 1) % len(modelos_ciclo)]
            # Cargar si no estaba cargado
            if modelo_actual in ("faster", "ambos") and faster_model is None:
                faster_model = load_faster_rcnn()
            if modelo_actual in ("yolo", "ambos") and yolo_model is None:
                yolo_model = load_yolo()
            print(f"\nModelo → {modelo_actual.upper()}")

        elif key == ord("+"):           # + → más confianza
            conf_thresh = min(0.95, conf_thresh + 0.05)
            print(f"\nConf → {conf_thresh:.2f}")

        elif key == ord("-"):           # - → menos confianza
            conf_thresh = max(0.05, conf_thresh - 0.05)
            print(f"\nConf → {conf_thresh:.2f}")

        frame_count += 1

    # ── Limpieza ─────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
        print(f"Video guardado: {os.path.abspath('demo_video.mp4')}")
    if display_ok:
        cv2.destroyAllWindows()

    if fps_list:
        print(f"\nResumen:")
        print(f"  Frames procesados : {frame_count}")
        print(f"  FPS promedio      : {np.mean(fps_list):.1f}")
        print(f"  FPS máximo        : {max(fps_list):.1f}")
        print(f"  FPS mínimo        : {min(fps_list):.1f}")


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detección en tiempo real — Práctica 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--modelo",   default="yolo",
                        choices=["yolo", "faster", "ambos"],
                        help="Modelo a usar (default: yolo)")
    parser.add_argument("--camara",   type=int, default=0,
                        help="Índice de cámara (default: 0)")
    parser.add_argument("--imgsz",    type=int, default=480,
                        help="Tamaño de imagen para inferencia (default: 480)")
    parser.add_argument("--conf",     type=float, default=0.4,
                        help="Umbral de confianza (default: 0.4)")
    parser.add_argument("--grabar",   action="store_true",
                        help="Grabar video demo y salir al terminar")
    parser.add_argument("--segundos", type=int, default=30,
                        help="Duración de grabación en segundos (default: 30)")

    args = parser.parse_args()
    run(args)