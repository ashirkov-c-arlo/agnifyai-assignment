from __future__ import annotations

import multiprocessing as mp
from multiprocessing import Process, Queue

import cv2
import numpy as np
from loguru import logger


def _paddle_worker(request_q: Queue, response_q: Queue, config: dict) -> None:
    """Worker process that owns PaddleOCR. Runs in isolation from TensorFlow."""
    try:
        from paddleocr import PaddleOCR

        import paddle

        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
            device=device,
            enable_mkldnn=False,
        )
    except Exception as exc:
        response_q.put(("init_error", str(exc)))
        return

    response_q.put(("init_ok", device))

    while True:
        msg = request_q.get()
        if msg is None:
            break
        try:
            image = msg
            if hasattr(ocr, "predict"):
                result = ocr.predict(image)
                texts, confidences = _parse_predict_result(result)
            else:
                result = ocr.ocr(image, cls=True)
                texts, confidences = _parse_legacy_result(result)
            response_q.put(("result", texts, confidences))
        except Exception as exc:
            response_q.put(("error", str(exc)))


def _parse_predict_result(result: list) -> tuple[list[str], list[float]]:
    texts = []
    confidences = []
    for page in result or []:
        page_result = (
            page.get("res", page)
            if isinstance(page, dict)
            else getattr(page, "json", {}).get("res", {})
        )
        texts.extend(str(text) for text in page_result.get("rec_texts", []))
        confidences.extend(float(score) for score in page_result.get("rec_scores", []))
    return texts, confidences


def _parse_legacy_result(result: list) -> tuple[list[str], list[float]]:
    texts = []
    confidences = []
    for page in result or []:
        for line in page or []:
            if len(line) < 2:
                continue
            text, confidence = line[1]
            texts.append(str(text))
            confidences.append(float(confidence))
    return texts, confidences


class ScoreboardOCR:
    """OCR engine for scoreboard text extraction. PaddleOCR runs in a subprocess."""

    def __init__(self, config: dict):
        self._config = config
        self._engine: str = "none"
        self._request_q: Queue | None = None
        self._response_q: Queue | None = None
        self._process: Process | None = None

        self._init_paddle_subprocess()

    def _init_paddle_subprocess(self) -> None:
        ctx = mp.get_context("spawn")
        self._request_q = ctx.Queue()
        self._response_q = ctx.Queue()

        self._process = ctx.Process(
            target=_paddle_worker,
            args=(self._request_q, self._response_q, self._config),
            daemon=True,
        )
        self._process.start()

        msg = self._response_q.get(timeout=120)
        if msg[0] == "init_ok":
            self._engine = "paddleocr"
            logger.debug("PaddleOCR subprocess started on device: {}", msg[1])
        else:
            self._engine = "none"
            logger.warning("PaddleOCR subprocess failed to start: {}", msg[1])
            self._cleanup()

    def read_text(self, crop: np.ndarray) -> tuple[str, float]:
        if crop is None or crop.size == 0:
            return ("", 0.0)

        if self._engine != "paddleocr":
            return ("", 0.0)

        processed = self._preprocess(crop)
        image = (
            cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            if len(processed.shape) == 2
            else processed
        )

        self._request_q.put(image)
        msg = self._response_q.get(timeout=30)

        if msg[0] == "result":
            texts, confidences = msg[1], msg[2]
            raw_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            return (raw_text, avg_confidence)

        logger.warning("PaddleOCR worker error: {}", msg[1])
        return ("", 0.0)

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        h, w = gray.shape[:2]
        if h < 50 or w < 100:
            scale = max(50 / h, 100 / w, 2.0)
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        return binary

    @property
    def engine_name(self) -> str:
        return self._engine

    @property
    def is_available(self) -> bool:
        return self._engine != "none"

    def _cleanup(self) -> None:
        if self._process and self._process.is_alive():
            self._request_q.put(None)
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
        self._process = None
        self._request_q = None
        self._response_q = None

    def __del__(self) -> None:
        self._cleanup()
