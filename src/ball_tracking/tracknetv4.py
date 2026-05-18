from __future__ import annotations

import math
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from src.schemas import BallDetection2D

INPUT_HEIGHT = 288
INPUT_WIDTH = 512


def _point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class TrackNetV4Detector:
    """TrackNetV4 ball detector backend."""

    def __init__(self, config: dict):
        tf = _import_tf()

        bt_config = config.get("ball_tracking", {})
        tn_config = bt_config.get("tracknetv4", {})

        self._input_width = int(tn_config.get("input_width", INPUT_WIDTH))
        self._input_height = int(tn_config.get("input_height", INPUT_HEIGHT))
        self._score_threshold = float(tn_config.get("score_threshold", 0.5))
        self._candidate_selection = str(tn_config.get("candidate_selection", "temporal_mass"))
        self._max_jump_px = float(tn_config.get("max_jump_px", 220.0))
        self._reacquire_gap_frames = int(tn_config.get("reacquire_gap_frames", 6))

        repo_src = _resolve_repo_src(tn_config)
        model_path = _resolve_model_path(config, tn_config)

        custom_objects = _get_custom_objects(repo_src)
        self._model = tf.keras.models.load_model(
            str(model_path), custom_objects=custom_objects
        )
        logger.info("TrackNetV4 model loaded from {}", model_path)

        self._recent_frames: deque[np.ndarray] = deque(maxlen=3)
        self._last_detection_xy: tuple[float, float] | None = None
        self._missing_frames = self._reacquire_gap_frames + 1

    def detect_frame(
        self, frame: np.ndarray, prev_frames: list[np.ndarray] | None = None
    ) -> BallDetection2D | None:
        if prev_frames is None:
            self._recent_frames.append(frame)
            if len(self._recent_frames) < 3:
                return None
            frames = list(self._recent_frames)
        else:
            frames = [*prev_frames, frame][-3:]
            if len(frames) < 3:
                return None

        orig_h, orig_w = frame.shape[:2]
        input_batch = self._preprocess(frames)
        predictions = self._model.predict(input_batch, batch_size=1, verbose=0)

        heatmap = predictions[0, 2]
        return self._postprocess(heatmap, orig_w, orig_h)

    def _preprocess(self, frames: list[np.ndarray]) -> np.ndarray:
        channels = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self._input_width, self._input_height))
            chw = np.moveaxis(resized, -1, 0).astype(np.float32)
            channels.append(chw)
        batch = np.concatenate(channels, axis=0)
        batch = batch / 255.0
        return batch.reshape(1, 9, self._input_height, self._input_width)

    def _postprocess(
        self, heatmap: np.ndarray, orig_w: int, orig_h: int
    ) -> BallDetection2D | None:
        candidates = self._extract_candidates(heatmap, orig_w, orig_h)
        if not candidates:
            return self._mark_missed()

        selected = self._select_candidate(candidates)
        if selected is None:
            return self._mark_missed()

        image_xy = selected["image_xy"]
        self._last_detection_xy = image_xy
        self._missing_frames = 0

        return BallDetection2D(
            frame=0,
            time_s=0.0,
            image_xy=image_xy,
            confidence=float(selected["confidence"]),
            visibility="visible",
            source="tracknetv4",
        )

    def _extract_candidates(
        self, heatmap: np.ndarray, orig_w: int, orig_h: int
    ) -> list[dict]:
        binary = (heatmap > self._score_threshold).astype(np.uint8)
        if binary.max() == 0:
            return []

        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []

        width_ratio = orig_w / self._input_width
        height_ratio = orig_h / self._input_height
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or h == 0:
                continue

            contour_mask = binary[y : y + h, x : x + w].astype(bool)
            if not np.any(contour_mask):
                continue

            heatmap_roi = heatmap[y : y + h, x : x + w]
            confidence = float(np.max(heatmap_roi[contour_mask]))
            mass = float(np.sum(heatmap_roi[contour_mask]))
            image_xy = ((x + w / 2.0) * width_ratio, (y + h / 2.0) * height_ratio)
            candidates.append(
                {
                    "image_xy": image_xy,
                    "area": w * h,
                    "confidence": confidence,
                    "mass": mass,
                }
            )

        return candidates

    def _select_candidate(self, candidates: list[dict]) -> dict | None:
        mode = self._candidate_selection
        if mode == "peak":
            return max(candidates, key=lambda c: c["confidence"])
        if mode == "mass":
            return max(candidates, key=lambda c: c["mass"])

        selection_pool = candidates
        if (
            mode.startswith("temporal")
            and self._last_detection_xy is not None
            and self._missing_frames <= self._reacquire_gap_frames
            and self._max_jump_px > 0
        ):
            selection_pool = []
            for candidate in candidates:
                dist = _point_distance(candidate["image_xy"], self._last_detection_xy)
                if dist <= self._max_jump_px:
                    candidate = dict(candidate)
                    candidate["distance_from_previous"] = dist
                    selection_pool.append(candidate)
            if not selection_pool:
                return None

        if mode == "temporal_mass":
            return max(selection_pool, key=self._temporal_mass_score)
        if mode == "temporal_area":
            return max(
                selection_pool,
                key=lambda c: (
                    c["area"],
                    c["confidence"],
                    -float(c.get("distance_from_previous", 0.0)),
                ),
            )
        return max(candidates, key=lambda c: c["area"])

    def _temporal_mass_score(self, candidate: dict) -> tuple[float, float]:
        dist = float(candidate.get("distance_from_previous", 0.0))
        if self._max_jump_px <= 0:
            distance_scale = 1.0
        else:
            distance_scale = 1.0 + dist / self._max_jump_px
        return (candidate["mass"] / distance_scale, candidate["confidence"])

    def _mark_missed(self) -> None:
        self._missing_frames += 1
        return None


def _import_tf():
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        return tf
    except ImportError as exc:
        raise RuntimeError(
            "TrackNetV4 backend requires tensorflow. Install with: pip install tensorflow"
        ) from exc


def _resolve_repo_src(tn_config: dict) -> Path:
    configured = tn_config.get("repo_path") or os.environ.get("TRACKNETV4_REPO")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([
        Path("third_party/TrackNetV4"),
        Path("external/TrackNetV4"),
        Path("../TrackNetV4"),
    ])

    for candidate in candidates:
        src = candidate / "src"
        if (src / "models" / "TrackNetV4.py").exists():
            return src.resolve()
        if (candidate / "models" / "TrackNetV4.py").exists():
            return candidate.resolve()

    raise RuntimeError(
        "TrackNetV4 repo not found; set ball_tracking.tracknetv4.repo_path or TRACKNETV4_REPO env"
    )


def _resolve_model_path(config: dict, tn_config: dict) -> Path:
    configured = tn_config.get("model_path")
    models_dir = Path(config.get("models", {}).get("cache_dir", "data/models"))

    candidates = []
    if configured:
        path = Path(configured).expanduser()
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend([Path.cwd() / path, models_dir / path])
    else:
        # Look in tracknetv4/ subdirectory first, then models_dir directly
        tracknetv4_dir = models_dir / "tracknetv4"
        for search_dir in [tracknetv4_dir, models_dir]:
            if search_dir.is_dir():
                keras_files = sorted(search_dir.glob("*.keras"))
                if keras_files:
                    return keras_files[0].resolve()
        candidates.append(tracknetv4_dir / "model.keras")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    display = configured if configured else "data/models/tracknetv4/*.keras or data/models/*.keras"
    raise RuntimeError(f"TrackNetV4 model weights not found: {display}")


def _get_custom_objects(repo_src: Path) -> dict:
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    try:
        from models.TrackNetV4 import (
            FusionLayerTypeA,
            FusionLayerTypeB,
            MotionPromptLayer,
            rearrange_tensor,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import TrackNetV4 model code from {repo_src}"
        ) from exc

    import tensorflow as tf
    from tensorflow.keras.layers import Layer

    # Patch MotionPromptLayer to accept extra kwargs and fix return order for saved weights
    _orig_init = MotionPromptLayer.__init__

    def _patched_init(self, penalty_weight=0.0, **kwargs):
        kwargs.pop("exp_name", None)
        _orig_init(self, penalty_weight=penalty_weight, **kwargs)

    def _patched_call(self, video_seq):
        loss = tf.constant(0.0)
        video_seq = rearrange_tensor(video_seq, self.input_permutation)
        norm_seq = video_seq * 0.225 + 0.45
        idx_list = [self.color_map[idx] for idx in self.input_color_order]
        gray_scale_tensor = tf.gather(self.gray_scale, idx_list)
        weights = tf.cast(gray_scale_tensor, dtype=norm_seq.dtype)
        grayscale_video_seq = tf.einsum("btcwh,c->btwh", norm_seq, weights)
        frame_diff = grayscale_video_seq[:, 1:] - grayscale_video_seq[:, :-1]
        attention_map = self.pn(frame_diff, self.a, self.b)
        return loss, attention_map

    MotionPromptLayer.__init__ = _patched_init
    MotionPromptLayer.call = _patched_call

    # Legacy layer names used in saved .keras weight files
    class MotionIncorporationLayerV1(Layer):
        def call(self, inputs):
            feature_map, attention_map = inputs
            o1 = feature_map[:, 0, :, :]
            o2 = feature_map[:, 1, :, :] * attention_map[:, 0, :, :]
            o3 = feature_map[:, 2, :, :] * attention_map[:, 1, :, :]
            return o1, o2, o3

    class MotionIncorporationLayerV2(Layer):
        def call(self, inputs):
            feature_map, attention_map = inputs
            o1 = feature_map[:, 0, :, :] * attention_map[:, 0, :, :]
            o2 = feature_map[:, 1, :, :] * (
                (attention_map[:, 0, :, :] + attention_map[:, 1, :, :]) / 2
            )
            o3 = feature_map[:, 2, :, :] * attention_map[:, 1, :, :]
            return o1, o2, o3

    class CombineOutputs(Layer):
        def call(self, inputs):
            return tf.stack(inputs, axis=1)

    return {
        "MotionPromptLayer": MotionPromptLayer,
        "FusionLayerTypeA": FusionLayerTypeA,
        "FusionLayerTypeB": FusionLayerTypeB,
        "MotionIncorporationLayerV1": MotionIncorporationLayerV1,
        "MotionIncorporationLayerV2": MotionIncorporationLayerV2,
        "CombineOutputs": CombineOutputs,
        "custom_loss": _custom_loss,
    }


def _custom_loss(y_true, y_pred):
    """TrackNetV4 focal-style loss (reproduced from upstream util.py)."""
    import tensorflow.keras.backend as K

    loss = (-1) * (
        K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1))
        + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1))
    )
    return loss
