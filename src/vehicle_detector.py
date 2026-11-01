"""Detecta, rastreia e conta veículos em um vídeo."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path
from tkinter import Tk, filedialog
from typing import DefaultDict, Deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# Paths and detector settings.
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "yolo11s.pt"
IMAGE_SIZE = 960
CONFIDENCE = 0.15
TRACKER_CONFIG = "bytetrack.yaml"
RECONNECT_INTERVAL_SECONDS = 2.0
VEHICLE_CLASSES = (2, 3, 5, 7)
CLASS_NAMES = {
    2: "Carro",
    3: "Moto",
    5: "Onibus",
    7: "Caminhao",
}


def obter_fonte() -> tuple[str | int, bool, int, float, str]:
    """Read camera/stream arguments or open the video file picker."""
    parser = argparse.ArgumentParser(
        description="Detecta e rastreia veículos em vídeo ou câmera ao vivo."
    )
    parser.add_argument("input", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument(
        "--source",
        help="Arquivo, índice da câmera (ex.: 0) ou URL de stream (ex.: RTSP).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=IMAGE_SIZE,
        help=f"Resolução de inferência (padrão: {IMAGE_SIZE}; maior ajuda em objetos distantes, mas reduz o FPS).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE,
        help=f"Confiança mínima entre 0 e 1 (padrão: {CONFIDENCE}).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Dispositivo: auto (padrão), cpu ou índice da GPU NVIDIA, como 0.",
    )
    args = parser.parse_args()

    if args.imgsz < 32:
        parser.error("--imgsz deve ser pelo menos 32.")
    if not 0 < args.conf <= 1:
        parser.error("--conf deve ser maior que 0 e no máximo 1.")
    if args.device.lower() == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    elif args.device.lower() == "cpu":
        device = "cpu"
    elif args.device.isdecimal():
        if not torch.cuda.is_available():
            parser.error(
                "GPU solicitada, mas o PyTorch instalado não tem CUDA disponível. "
                "Instale uma versão CUDA do PyTorch compatível com sua GPU e driver."
            )
        device = args.device
    else:
        parser.error("--device aceita auto, cpu ou o índice de uma GPU NVIDIA (ex.: 0).")

    if args.source and args.input:
        parser.error("Informe a fonte por --source ou como argumento posicional, não ambos.")
    requested_source = args.source or args.input
    if requested_source is None:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            selected = filedialog.askopenfilename(
                title="Selecione o vídeo ou stream gravado",
                filetypes=[
                    ("Vídeos", "*.mp4 *.avi *.mov *.mkv *.wmv *.mpeg *.mpg"),
                    ("Todos os arquivos", "*.*"),
                ],
            )
        finally:
            root.destroy()
        if not selected:
            return "", False, args.imgsz, args.conf, device
        return str(Path(selected).expanduser()), False, args.imgsz, args.conf, device

    source = requested_source.strip()
    if source.isdecimal():
        return int(source), True, args.imgsz, args.conf, device
    if source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return source, True, args.imgsz, args.conf, device

    path = Path(source).expanduser()
    if path.is_file():
        return str(path), False, args.imgsz, args.conf, device
    print(f"Fonte não encontrada: {path}")
    return "", False, args.imgsz, args.conf, device


def abrir_fonte(source: str | int) -> cv2.VideoCapture:
    """Open a local camera, network stream, or video file."""
    if isinstance(source, int):
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    elif source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        capture = cv2.VideoCapture(source)

    if capture.isOpened():
        # Reduce buffering where supported so live overlays stay near real time.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def desenhar_painel(frame: np.ndarray, counts: dict[str, set[int]], fps: float) -> None:
    """Draw summary counts over the current video frame."""
    total = sum(len(vehicle_ids) for vehicle_ids in counts.values())
    lines = [
        f"FPS: {fps:.1f}",
        f"Total de veiculos: {total}",
        f"Carros: {len(counts['Carro'])}",
        f"Motos: {len(counts['Moto'])}",
        f"Onibus: {len(counts['Onibus'])}",
        f"Caminhoes: {len(counts['Caminhao'])}",
    ]
    for line_number, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (15, 30 + line_number * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )


def imprimir_relatorio(vehicle_counts: dict[str, set[int]]) -> None:
    """Print the final vehicle totals by category."""
    total = sum(len(vehicle_ids) for vehicle_ids in vehicle_counts.values())
    print("\n========== RELATORIO FINAL ==========")
    print(f"Total de veiculos: {total}")
    print("\nPor categoria:")
    for category in CLASS_NAMES.values():
        print(f"{category}: {len(vehicle_counts[category])}")
    print("\n=====================================")


def iniciar_detector() -> None:
    """Run vehicle detection on a file, local camera, or network stream."""
    source, is_live, image_size, confidence, device = obter_fonte()
    if source == "":
        print("Nenhuma fonte selecionada. Encerrando.")
        return

    if not MODEL_PATH.is_file():
        print(f"Modelo YOLO não encontrado: {MODEL_PATH}")
        return

    source_label = "câmera/stream ao vivo" if is_live else "vídeo"
    print(f"\nFonte selecionada ({source_label}).")
    if device == "cpu":
        print("Dispositivo de inferência: CPU.")
    else:
        print(f"Dispositivo de inferência: GPU {device} ({torch.cuda.get_device_name(int(device))}).")
    print("Carregando modelo YOLO...")
    model = YOLO(str(MODEL_PATH))
    capture = abrir_fonte(source)
    if not capture.isOpened():
        capture.release()
        print("Não foi possível abrir a fonte. Verifique o índice, URL e conexão.")
        return

    vehicle_counts: DefaultDict[str, set[int]] = defaultdict(set)
    fps_history: Deque[float] = deque(maxlen=30)
    previous_time = time.perf_counter()
    track_id_offset = 0
    window_name = "SIRV - Monitoramento de Veículos"

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            success, frame = capture.read()
            if not success:
                if not is_live:
                    print("\nFim do vídeo ou falha na leitura.")
                    break

                print("\nSinal da câmera perdido. Tentando reconectar...")
                capture.release()
                while True:
                    key = cv2.waitKey(int(RECONNECT_INTERVAL_SECONDS * 1000)) & 0xFF
                    if key in (27, ord("q")):
                        return
                    capture = abrir_fonte(source)
                    if capture.isOpened():
                        print("Câmera reconectada.")
                        break
                    capture.release()
                    print("Reconexão falhou. Nova tentativa em 2 segundos; pressione Q para sair.")

                # Start a fresh tracker and keep IDs unique in the report.
                model = YOLO(str(MODEL_PATH))
                existing_ids = [
                    vehicle_id
                    for ids in vehicle_counts.values()
                    for vehicle_id in ids
                ]
                track_id_offset = max(existing_ids, default=-1) + 1
                continue

            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CONFIG,
                imgsz=image_size,
                conf=confidence,
                iou=0.45,
                classes=list(VEHICLE_CLASSES),
                device=device,
                verbose=False,
            )
            boxes = results[0].boxes
            if boxes is not None:
                coordinates = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                track_ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if boxes.id is not None
                    else [None] * len(coordinates)
                )

                for box, class_id, vehicle_id in zip(coordinates, classes, track_ids):
                    category = CLASS_NAMES.get(class_id, "Desconhecido")
                    if vehicle_id is not None:
                        vehicle_id = int(vehicle_id) + track_id_offset
                        vehicle_counts[category].add(vehicle_id)
                        label = f"ID {vehicle_id} - {category}"
                    else:
                        label = category

                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time
            if elapsed > 0:
                fps_history.append(1.0 / elapsed)
            fps = sum(fps_history) / len(fps_history) if fps_history else 0.0
            desenhar_painel(frame, vehicle_counts, fps)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("\nExecução interrompida pelo usuário.")
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    imprimir_relatorio(vehicle_counts)


if __name__ == "__main__":
    iniciar_detector()



