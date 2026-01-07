import io
import os
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from backend.capture_model import check_quality, save_capture

# Prefer the lightweight tflite_runtime; fall back to tensorflow-lite interpreter if available.
try:  # pragma: no cover - runtime environment dependent
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except Exception:  # pragma: no cover
    try:
        from tensorflow.lite import Interpreter  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "No TFLite interpreter found. Install tflite-runtime or tensorflow to run inference."
        ) from exc

app = FastAPI(title="Dog Nose Frame Detector", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get(
    "DOG_NOSE_TFLITE_PATH",
    os.path.join(os.path.dirname(__file__), "models", "dog_nose_detector.tflite"),
)
# Location to store curated dataset samples
DATASET_DIR = Path(
    os.environ.get(
        "DOG_NOSE_DATASET_DIR",
        Path(__file__).resolve().parent / "dataset",
    )
)
ROI_FRACTION = float(os.environ.get("DOG_NOSE_ROI_FRACTION", 0.5))
MIN_LAPLACIAN = float(os.environ.get("DOG_NOSE_MIN_LAPLACIAN", 150.0))
MIN_TENENGRAD = float(os.environ.get("DOG_NOSE_MIN_TENENGRAD", 200.0))
MIN_CONTRAST = float(os.environ.get("DOG_NOSE_MIN_CONTRAST", 60.0))
MIN_BRIGHTNESS = float(os.environ.get("DOG_NOSE_MIN_BRIGHTNESS", 50.0))
MAX_BRIGHTNESS = float(os.environ.get("DOG_NOSE_MAX_BRIGHTNESS", 200.0))
# Frame region (normalized 0-1) that represents the on-screen guide.
FRAME_BOUNDS = {
    "xmin": float(os.environ.get("DOG_NOSE_FRAME_XMIN", 0.2)),
    "xmax": float(os.environ.get("DOG_NOSE_FRAME_XMAX", 0.8)),
    "ymin": float(os.environ.get("DOG_NOSE_FRAME_YMIN", 0.25)),
    "ymax": float(os.environ.get("DOG_NOSE_FRAME_YMAX", 0.85)),
}
MIN_SCORE = float(os.environ.get("DOG_NOSE_MIN_SCORE", 0.55))
MIN_OVERLAP = float(os.environ.get("DOG_NOSE_MIN_OVERLAP", 0.5))
LOG_LIMIT = int(os.environ.get("DOG_NOSE_LOG_LIMIT", 200))

log_buffer: Deque[Dict[str, object]] = deque(maxlen=LOG_LIMIT)


def _log(source: str, message: str, extra: Optional[Dict[str, object]] = None) -> None:
    entry: Dict[str, object] = {
        "ts": time.time(),
        "source": source,
        "message": message,
    }
    if extra:
        entry.update(extra)
    log_buffer.append(entry)


@lru_cache(maxsize=1)
def get_interpreter() -> Interpreter:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"TFLite model not found at {MODEL_PATH}")
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=2)
    interpreter.allocate_tensors()
    return interpreter


def _load_image(file: UploadFile) -> Image.Image:
    try:
        data = file.file.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail="Invalid image upload") from exc


def _preprocess(image: Image.Image, input_shape: Tuple[int, int, int, int]) -> np.ndarray:
    _, height, width, _ = input_shape
    resized = image.resize((width, height))
    array = np.array(resized, dtype=np.float32)
    # Normalize to 0-1; adjust if your model expects something else.
    array = array / 255.0
    return np.expand_dims(array, axis=0)


def _run_inference(img: Image.Image) -> Dict[str, np.ndarray]:
    interpreter = get_interpreter()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    tensor = _preprocess(img, input_details[0]["shape"])  # type: ignore[index]
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()

    outputs: Dict[str, np.ndarray] = {}
    for detail in output_details:
        name = detail.get("name", str(detail["index"]))
        outputs[name] = interpreter.get_tensor(detail["index"])
    return outputs


def _extract_detection(outputs: Dict[str, np.ndarray]) -> Tuple[Optional[Tuple[float, float, float, float]], float]:
    # Try common detection output names/order; adjust to your model signature.
    boxes = None
    scores = None
    for key in outputs.keys():
        low_key = key.lower()
        if boxes is None and "box" in low_key:
            boxes = outputs[key]
        if scores is None and ("score" in low_key or "prob" in low_key):
            scores = outputs[key]
    if boxes is None or scores is None:
        return None, 0.0

    boxes = np.squeeze(boxes, axis=0)
    scores = np.squeeze(scores, axis=0)
    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, 0)
    if scores.ndim == 0:
        scores = np.expand_dims(scores, 0)

    idx = int(np.argmax(scores))
    score = float(scores[idx])
    box = boxes[idx]
    # Expect box as [ymin, xmin, ymax, xmax]; coerce length.
    if box.size >= 4:
        ymin, xmin, ymax, xmax = map(float, box[:4])
        return (ymin, xmin, ymax, xmax), score
    return None, score


def _overlap_ratio(box: Tuple[float, float, float, float], frame: Dict[str, float]) -> float:
    ymin, xmin, ymax, xmax = box
    ixmin = max(xmin, frame["xmin"])
    iymin = max(ymin, frame["ymin"])
    ixmax = min(xmax, frame["xmax"])
    iymax = min(ymax, frame["ymax"])
    inter_w = max(0.0, ixmax - ixmin)
    inter_h = max(0.0, iymax - iymin)
    inter_area = inter_w * inter_h
    box_area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
    if box_area == 0:
        return 0.0
    return inter_area / box_area


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _normalized_box_to_px(
    box: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = box
    x1 = max(0, min(int(xmin * width), width - 1))
    y1 = max(0, min(int(ymin * height), height - 1))
    x2 = max(x1 + 1, min(int(xmax * width), width))
    y2 = max(y1 + 1, min(int(ymax * height), height))
    return x1, y1, x2, y2


def _center_square_roi(width: int, height: int) -> Tuple[int, int, int, int]:
    side = int(min(width, height) * ROI_FRACTION)
    side = max(1, side)
    cx, cy = width // 2, height // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)
    return x1, y1, x2, y2


def _next_capture_id(base_dir: Path) -> int:
    original_dir = base_dir / "original"
    original_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in original_dir.glob("nose_*.jpg")]
    return len(existing) + 1


class ClientLog(BaseModel):
    level: str
    message: str
    meta: Optional[Dict[str, object]] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/detect-nose")
def detect_nose(file: UploadFile = File(...)) -> Dict[str, object]:
    image = _load_image(file)

    if not os.path.exists(MODEL_PATH):
        reason = f"TFLite model not found at {MODEL_PATH}"
        _log("backend", "model_missing", {"reason": reason})
        return {
            "capturable": False,
            "score": 0.0,
            "reason": reason,
            "frame": FRAME_BOUNDS,
            "min_score": MIN_SCORE,
            "min_overlap": MIN_OVERLAP,
        }

    try:
        outputs = _run_inference(image)
    except RuntimeError as exc:
        reason = str(exc)
        _log("backend", "model_missing", {"reason": reason})
        return {
            "capturable": False,
            "score": 0.0,
            "reason": reason,
            "frame": FRAME_BOUNDS,
            "min_score": MIN_SCORE,
            "min_overlap": MIN_OVERLAP,
        }

    box, score = _extract_detection(outputs)

    if box is None:
        result = {"capturable": False, "score": score, "reason": "No detection output"}
        _log("backend", "no detection output", {"score": score})
        return result

    overlap = _overlap_ratio(box, FRAME_BOUNDS)
    capturable = bool(score >= MIN_SCORE and overlap >= MIN_OVERLAP)
    result = {
        "capturable": capturable,
        "score": round(score, 4),
        "box": {
            "ymin": round(box[0], 4),
            "xmin": round(box[1], 4),
            "ymax": round(box[2], 4),
            "xmax": round(box[3], 4),
        },
        "frame": FRAME_BOUNDS,
        "overlap": round(overlap, 4),
        "min_score": MIN_SCORE,
        "min_overlap": MIN_OVERLAP,
    }
    _log(
        "backend",
        "detection",
        {
          "capturable": capturable,
          "score": result["score"],
          "overlap": result["overlap"],
          "box": result["box"],
        },
    )
    return result


@app.post("/dataset/capture")
def capture_dataset_sample(file: UploadFile = File(...)) -> Dict[str, object]:
    image = _load_image(file)

    if not os.path.exists(MODEL_PATH):
        reason = f"TFLite model not found at {MODEL_PATH}"
        _log("backend", "dataset_capture_model_missing", {"reason": reason})
        raise HTTPException(status_code=503, detail=reason)

    width, height = image.size
    frame_bgr = _pil_to_bgr(image)

    try:
        outputs = _run_inference(image)
    except RuntimeError as exc:
        reason = str(exc)
        _log("backend", "dataset_capture_inference_error", {"reason": reason})
        raise HTTPException(status_code=500, detail=reason)

    box, score = _extract_detection(outputs)
    overlap = 0.0
    capturable = False
    roi_coords: Tuple[int, int, int, int] = _center_square_roi(width, height)

    if box is not None:
        overlap = _overlap_ratio(box, FRAME_BOUNDS)
        capturable = bool(score >= MIN_SCORE and overlap >= MIN_OVERLAP)
        roi_coords = _normalized_box_to_px(box, width, height)

    x1, y1, x2, y2 = roi_coords
    roi_gray = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    quality = check_quality(
        roi_gray,
        min_laplacian=MIN_LAPLACIAN,
        min_tenengrad=MIN_TENENGRAD,
        min_contrast=MIN_CONTRAST,
        min_brightness=MIN_BRIGHTNESS,
        max_brightness=MAX_BRIGHTNESS,
    )

    should_save = bool(capturable and quality["passed"])
    saved_path: Optional[str] = None

    if should_save:
        capture_id = _next_capture_id(DATASET_DIR)
        saved = save_capture(frame_bgr, roi_coords, quality, DATASET_DIR, capture_id)
        saved_path = str(saved.relative_to(DATASET_DIR)) if saved.exists() else str(saved)
        _log(
            "backend",
            "dataset_capture_saved",
            {
                "capture_id": capture_id,
                "path": saved_path,
                "score": round(score, 4),
                "overlap": round(overlap, 4),
                "quality": quality,
            },
        )
    else:
        _log(
            "backend",
            "dataset_capture_rejected",
            {
                "capturable": capturable,
                "quality_passed": quality["passed"],
                "score": round(score, 4),
                "overlap": round(overlap, 4),
            },
        )

    return {
        "saved": should_save,
        "capturable": capturable,
        "score": round(score, 4),
        "overlap": round(overlap, 4),
        "box": {
            "ymin": round(box[0], 4) if box else None,
            "xmin": round(box[1], 4) if box else None,
            "ymax": round(box[2], 4) if box else None,
            "xmax": round(box[3], 4) if box else None,
        },
        "frame": FRAME_BOUNDS,
        "quality": quality,
        "saved_path": saved_path,
        "reason": None if should_save else "Detection/quality thresholds not met",
    }


@app.get("/")
def root() -> Dict[str, object]:
    return {
        "service": "Dog Nose Frame Detector",
        "endpoints": {"detect": "/detect-nose", "health": "/health", "capture": "/dataset/capture"},
        "model_path": MODEL_PATH,
        "frame": FRAME_BOUNDS,
        "min_score": MIN_SCORE,
        "min_overlap": MIN_OVERLAP,
    }


@app.get("/logs")
def get_logs() -> Dict[str, object]:
    return {"logs": list(log_buffer)}


@app.post("/client-log")
def client_log(payload: ClientLog) -> Dict[str, str]:
    _log("frontend", payload.message, {"level": payload.level, "meta": payload.meta})
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=bool(int(os.environ.get("RELOAD", "1"))),
    )
