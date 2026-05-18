from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from src.ball_tracking.kalman_tracker import BallKalmanTracker
from src.schemas import BallDetection2D, BallTrack2D, CourtRegistration2D
from src.video_io.reader import get_video_info


def track_ball(
    video_path: str,
    config: dict,
    registration: CourtRegistration2D | None = None,
) -> tuple[list[BallDetection2D], list[BallTrack2D]]:
    """Compatibility wrapper for the single-pass ball detector and Kalman tracker."""
    from src.ball_tracking.ball_processor import BallFrameProcessor
    from src.video_io.single_pass import run_single_pass

    info = get_video_info(video_path)
    fps = info["fps"]
    total_frames = info["total_frames"]

    court_roi = _build_court_roi(registration, info) if registration else None
    court_roi = _validate_court_roi(court_roi, registration, info, config)

    processor = BallFrameProcessor(config, fps, court_roi)
    run_single_pass(video_path, [processor])
    detections = processor.detections
    tracks = build_ball_tracks(detections, config, total_frames, fps)

    logger.info(
        "Ball tracking complete: {} detections, {} track frames",
        len(detections),
        len(tracks),
    )
    return detections, tracks


def build_ball_tracks(
    detections: list[BallDetection2D],
    config: dict,
    total_frames: int,
    fps: float,
) -> list[BallTrack2D]:
    """Run Kalman tracking on pre-computed ball detections.

    This is the post-processing step that can be used after single-pass detection.
    """
    bt_config = config.get("ball_tracking", {})
    max_gap = int(bt_config.get("max_gap_frames", 10))
    max_interpolation_step_px = float(bt_config.get("max_interpolation_step_px", 220.0))
    process_noise = bt_config.get("kalman_process_noise", 0.1)
    gate_cfg = bt_config.get("kalman_gate", {})

    det_by_frame: dict[int, BallDetection2D] = {d.frame: d for d in detections}

    def new_kalman() -> BallKalmanTracker:
        return BallKalmanTracker(
            process_noise=process_noise,
            gate_threshold=float(gate_cfg.get("threshold", 0.0)),
            min_pixel_radius=float(gate_cfg.get("min_pixel_radius", 0.0)),
            q_gap_factor=float(gate_cfg.get("q_gap_factor", 0.0)),
            q_speed_factor=float(gate_cfg.get("q_speed_factor", 0.0)),
        )

    kalman = new_kalman()
    accepted_tracks: list[BallTrack2D] = []
    gap_count = 0

    for frame_idx in range(total_frames):
        time_s = frame_idx / fps if fps > 0 else 0.0
        det = det_by_frame.get(frame_idx)

        if det is not None:
            if kalman.initialized and gap_count > max_gap:
                kalman = new_kalman()
            elif kalman.initialized:
                kalman.predict(gap_len=gap_count)

            accepted = kalman.update(det.image_xy, det.confidence)

            if accepted:
                _, vel = kalman.get_state()
                velocity_px_s = (vel[0] * fps, vel[1] * fps) if fps > 0 else None

                accepted_tracks.append(
                    BallTrack2D(
                        frame=frame_idx,
                        time_s=time_s,
                        image_xy=det.image_xy,
                        velocity_px_s=velocity_px_s,
                        confidence=det.confidence,
                        state="tracked",
                        interpolated=False,
                        gap_len=0,
                    )
                )
                gap_count = 0
                continue

            # Detection rejected by gate — treat as gap (predict already called)
            gap_count += 1
            continue

        if kalman.initialized:
            gap_count += 1
            if gap_count <= max_gap:
                kalman.predict(gap_len=gap_count)

    return _fill_ball_track_gaps(
        accepted_tracks=accepted_tracks,
        total_frames=total_frames,
        fps=fps,
        max_gap=max_gap,
        max_interpolation_step_px=max_interpolation_step_px,
    )


def _fill_ball_track_gaps(
    accepted_tracks: list[BallTrack2D],
    total_frames: int,
    fps: float,
    max_gap: int,
    max_interpolation_step_px: float,
) -> list[BallTrack2D]:
    if not accepted_tracks:
        return []

    tracks: list[BallTrack2D] = [accepted_tracks[0]]

    for previous, current in zip(accepted_tracks, accepted_tracks[1:]):
        gap = current.frame - previous.frame - 1
        if gap > 0:
            if _should_interpolate_gap(previous, current, gap, max_gap, max_interpolation_step_px):
                tracks.extend(_interpolate_ball_gap(previous, current, gap, fps))
            else:
                tracks.extend(_missing_ball_gap(previous, current.frame, fps))

        tracks.append(current)

    tracks.extend(_missing_ball_tail(accepted_tracks[-1], total_frames, fps))

    return tracks


def _should_interpolate_gap(
    previous: BallTrack2D,
    current: BallTrack2D,
    gap: int,
    max_gap: int,
    max_interpolation_step_px: float,
) -> bool:
    if gap > max_gap:
        return False

    dx = current.image_xy[0] - previous.image_xy[0]
    dy = current.image_xy[1] - previous.image_xy[1]
    step_px = float(np.hypot(dx, dy)) / (gap + 1)
    return step_px <= max_interpolation_step_px


def _interpolate_ball_gap(
    previous: BallTrack2D,
    current: BallTrack2D,
    gap: int,
    fps: float,
) -> list[BallTrack2D]:
    dx = current.image_xy[0] - previous.image_xy[0]
    dy = current.image_xy[1] - previous.image_xy[1]
    velocity_px_s = (dx * fps / (gap + 1), dy * fps / (gap + 1)) if fps > 0 else None
    confidence = max(0.1, min(previous.confidence, current.confidence) * 0.75)

    interpolated = []
    for gap_idx in range(1, gap + 1):
        alpha = gap_idx / (gap + 1)
        frame_idx = previous.frame + gap_idx
        time_s = frame_idx / fps if fps > 0 else 0.0
        interpolated.append(
            BallTrack2D(
                frame=frame_idx,
                time_s=time_s,
                image_xy=(
                    previous.image_xy[0] + dx * alpha,
                    previous.image_xy[1] + dy * alpha,
                ),
                velocity_px_s=velocity_px_s,
                confidence=confidence,
                state="interpolated",
                interpolated=True,
                gap_len=gap_idx,
            )
        )

    return interpolated


def _missing_ball_gap(
    previous: BallTrack2D,
    next_frame: int,
    fps: float,
) -> list[BallTrack2D]:
    missing = []
    for frame_idx in range(previous.frame + 1, next_frame):
        gap_len = frame_idx - previous.frame
        time_s = frame_idx / fps if fps > 0 else 0.0
        missing.append(
            BallTrack2D(
                frame=frame_idx,
                time_s=time_s,
                image_xy=previous.image_xy,
                velocity_px_s=None,
                confidence=0.0,
                state="missing",
                interpolated=False,
                gap_len=gap_len,
            )
        )
    return missing


def _missing_ball_tail(
    previous: BallTrack2D,
    total_frames: int,
    fps: float,
) -> list[BallTrack2D]:
    missing = []
    for frame_idx in range(previous.frame + 1, total_frames):
        gap_len = frame_idx - previous.frame
        time_s = frame_idx / fps if fps > 0 else 0.0
        missing.append(
            BallTrack2D(
                frame=frame_idx,
                time_s=time_s,
                image_xy=previous.image_xy,
                velocity_px_s=None,
                confidence=0.0,
                state="missing",
                interpolated=False,
                gap_len=gap_len,
            )
        )
    return missing


def _validate_court_roi(
    mask: np.ndarray | None,
    registration: CourtRegistration2D | None,
    video_info: dict,
    config: dict,
) -> np.ndarray | None:
    """Drop court ROI if registration quality is too low or mask is too small."""
    if mask is None:
        return None

    roi_cfg = config.get("ball_tracking", {}).get("court_roi", {})
    mode = str(roi_cfg.get("mode", "auto"))

    if mode == "none":
        return None
    if mode == "court":
        return mask

    # mode == "auto": check quality
    min_confidence = float(roi_cfg.get("min_confidence", 0.5))
    min_area_pct = float(roi_cfg.get("min_area_pct", 35.0))

    if registration is not None and registration.confidence < min_confidence:
        logger.info(
            "Court ROI disabled: registration confidence {:.2f} < {:.2f}",
            registration.confidence,
            min_confidence,
        )
        return None

    h = video_info["height"]
    w = video_info["width"]
    mask_pct = float(np.count_nonzero(mask)) / (h * w) * 100.0
    if mask_pct < min_area_pct:
        logger.info(
            "Court ROI disabled: mask area {:.1f}% < {:.1f}%",
            mask_pct,
            min_area_pct,
        )
        return None

    return mask


def _build_court_roi(
    registration: CourtRegistration2D, video_info: dict
) -> np.ndarray | None:
    """Build a binary mask covering the court area in image space."""
    if registration.mode == "pixel_only" or registration.homography_court_to_image is None:
        return None

    h = video_info["height"]
    w = video_info["width"]

    # Project court corners to image space
    H = np.array(registration.homography_court_to_image, dtype=np.float64)

    # Court corners in court coords (10m x 20m)
    court_corners = np.array(
        [[0, 0], [10, 0], [10, 20], [0, 20]], dtype=np.float64
    )

    # Project to image
    ones = np.ones((4, 1), dtype=np.float64)
    pts_h = np.hstack([court_corners, ones])
    projected = (H @ pts_h.T).T
    projected_xy = projected[:, :2] / projected[:, 2:3]

    # Create mask with some margin
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = projected_xy.astype(np.int32).reshape((-1, 1, 2))

    # Expand polygon slightly for margin
    center = pts.mean(axis=0)
    expanded = ((pts - center) * 1.2 + center).astype(np.int32)

    cv2.fillPoly(mask, [expanded], 255)
    return mask
