"""Lightweight kinematic tracker for filling in gaps between YOLO detections.

YOLO detects the ball on maybe 60-90% of frames; the rest have motion blur,
occlusion, or brief out-of-frame moments where the model legitimately
cannot fire. Without help, our overlay flickers on/off as a result.

BallTracker holds (position, velocity, frames-since-detection) and:
- accepts a real detection, smoothing velocity from the gap
- rejects implausible jumps (false positive on a different object)
- extrapolates forward during gaps using last known velocity
- gives up after max_extrapolate_frames, so a truly absent ball stops
  drawing instead of drifting off-screen

This is a simple constant-velocity predictor — no Kalman filter. Good
enough for visual continuity; not pretending to be high-precision tracking.

Returns shape per frame:
    (centroid, is_real_detection) or None if we should not draw at all
"""
from __future__ import annotations


class BallTracker:
    def __init__(
        self,
        max_jump_per_frame: int = 150,
        max_extrapolate_frames: int = 8,
        velocity_smoothing: float = 0.6,
    ):
        # State
        self.last_centroid: tuple[int, int] | None = None
        self.velocity: tuple[float, float] = (0.0, 0.0)
        self.frames_since_detection: int = 0

        # Config
        self.max_jump_per_frame = max_jump_per_frame
        self.max_extrapolate_frames = max_extrapolate_frames
        self.velocity_smoothing = velocity_smoothing  # 0=fully reactive, 1=ignore new data

    def feed(
        self, detection: tuple[int, int] | None
    ) -> tuple[tuple[int, int], bool] | None:
        """Feed this frame's detection (centroid or None).

        Returns (centroid, is_real_detection) — is_real_detection is True if
        the detection was accepted, False if we're extrapolating during a gap.
        Returns None when we should not draw (no state yet, or too many
        consecutive missed frames).
        """
        if detection is None:
            return self._extrapolate()

        # Jump check: allowed distance scales with the gap. If YOLO missed
        # N frames, a real ball could legitimately have moved N*max_jump_per_frame.
        if self.last_centroid is not None:
            allowed = self.max_jump_per_frame * (1 + self.frames_since_detection)
            dx = detection[0] - self.last_centroid[0]
            dy = detection[1] - self.last_centroid[1]
            if dx * dx + dy * dy > allowed * allowed:
                # Treat this detection as a false positive; fall through to
                # extrapolation. (We don't discard our state — the real ball
                # might re-appear nearby on a later frame.)
                return self._extrapolate()

        # Accept the detection. Update velocity smoothed against the gap.
        if self.last_centroid is not None:
            gap = max(1, self.frames_since_detection + 1)
            dx_per_frame = (detection[0] - self.last_centroid[0]) / gap
            dy_per_frame = (detection[1] - self.last_centroid[1]) / gap
            s = self.velocity_smoothing
            self.velocity = (
                s * self.velocity[0] + (1 - s) * dx_per_frame,
                s * self.velocity[1] + (1 - s) * dy_per_frame,
            )

        self.last_centroid = detection
        self.frames_since_detection = 0
        return (detection, True)

    def _extrapolate(self) -> tuple[tuple[int, int], bool] | None:
        self.frames_since_detection += 1
        if self.last_centroid is None:
            return None
        if self.frames_since_detection > self.max_extrapolate_frames:
            return None
        cx = int(self.last_centroid[0] + self.velocity[0] * self.frames_since_detection)
        cy = int(self.last_centroid[1] + self.velocity[1] * self.frames_since_detection)
        return ((cx, cy), False)
