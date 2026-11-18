from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from moviepy.Clip import Clip
from moviepy.Effect import Effect


@dataclass
class RollingShutter(Effect):
    """Simulate the rolling shutter of a CMOS camera sensor.

    Instead of exposing the whole frame at once, a rolling-shutter sensor is
    read out row by row, so every row of the output frame is sampled from a
    slightly different time of the source clip. Fast-moving objects therefore
    appear slanted, stretched or wobbling ("jello effect").

    Parameters
    ----------
    readout_duration
        Total time (in seconds) the sensor takes to read the whole frame.
        The middle row of each output frame is sampled at the output time
        ``t``, the first scanned row at ``t - readout_duration / 2`` and the
        last scanned row at ``t + readout_duration / 2``. Must be >= 0. A
        readout duration of 0 disables the effect and the clip is returned
        unchanged.

    direction
        Scan direction of the sensor: ``"top_to_bottom"`` (default) or
        ``"bottom_to_top"``.

    out_of_bounds
        Policy for rows whose sampling time falls outside the source clip
        (e.g. negative times at the start of the clip, or times past its
        duration at the end):

        - ``"freeze"`` (default): clamp the sampling time, i.e. extend the
          first/last frame of the clip.
        - ``"transparent"``: make these rows transparent (black in the RGB
          layers, 0 in the mask). A mask is added to the clip if it has none.

    The effect requires a clip with a fixed ``fps`` and a set ``duration``,
    and frames of constant size. Output size, duration, audio and mask are
    preserved. Source frames are sampled at the clip's frame granularity and
    cached, so repeated reads of the same source frame are cheap.
    """

    readout_duration: float
    direction: str = "top_to_bottom"
    out_of_bounds: str = "freeze"

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        if self.direction not in ("top_to_bottom", "bottom_to_top"):
            raise ValueError(
                "direction must be 'top_to_bottom' or 'bottom_to_top', "
                f"got {self.direction!r}"
            )
        if self.out_of_bounds not in ("freeze", "transparent"):
            raise ValueError(
                "out_of_bounds must be 'freeze' or 'transparent', "
                f"got {self.out_of_bounds!r}"
            )
        if self.readout_duration < 0:
            raise ValueError(
                f"readout_duration must be >= 0, got {self.readout_duration}"
            )
        if self.readout_duration == 0:
            # No readout time: every row is sampled at t, effect is identity.
            return clip
        fps = getattr(clip, "fps", None)
        if fps is None:
            raise ValueError("RollingShutter requires a clip with a fixed fps")
        if clip.duration is None:
            raise ValueError("Attribute 'duration' not set")
        n_frames = max(1, int(round(clip.duration * fps)))
        height = clip.size[1]

        # Relative sampling time of each row, centered on the output time.
        if height > 1:
            fracs = np.arange(height) / (height - 1)
        else:
            fracs = np.array([0.5])
        if self.direction == "bottom_to_top":
            fracs = 1.0 - fracs
        offsets = (fracs - 0.5) * self.readout_duration

        freeze = self.out_of_bounds == "freeze"

        # Cache of recently sampled source frames (a sliding window is enough,
        # consecutive output frames read overlapping ranges of source frames).
        max_cached = int(np.ceil(self.readout_duration * fps)) + 4

        def make_filter():
            # One filter (with its own cache and shape check) per transformed
            # clip: the RGB clip and its mask have different frame shapes.
            cache = OrderedDict()
            expected_shape = []

            def get_source_frame(get_frame, k):
                try:
                    frame = cache[k]
                    cache.move_to_end(k)
                    return frame
                except KeyError:
                    pass
                frame = get_frame(k / fps)
                if not expected_shape:
                    expected_shape.append(frame.shape)
                elif frame.shape != expected_shape[0]:
                    raise ValueError(
                        "RollingShutter does not support clips with "
                        "variable-size frames (got "
                        f"{frame.shape}, expected {expected_shape[0]})"
                    )
                cache[k] = frame
                while len(cache) > max_cached:
                    cache.popitem(last=False)
                return frame

            def filter(get_frame, t):
                # Source frame index sampled by each row.
                ks = np.rint((t + offsets) * fps).astype(np.int64)
                valid = (ks >= 0) & (ks < n_frames)
                ks_clipped = np.clip(ks, 0, n_frames - 1)

                out = None
                for k in np.unique(ks_clipped):
                    frame = get_source_frame(get_frame, int(k))
                    if out is None:
                        # 'freeze' writes every row; 'transparent' leaves
                        # out-of-bounds rows at 0 (black / transparent).
                        out = (np.empty if freeze else np.zeros)(
                            frame.shape, dtype=frame.dtype
                        )
                    rows = ks_clipped == k
                    if not freeze:
                        rows = rows & valid
                    out[rows] = frame[rows]
                return out

            return filter

        new_clip = clip.transform(make_filter())
        if not freeze and new_clip.mask is None:
            # Transparency needs a mask to be visible. It is added after the
            # transform so that composite sources are left untouched.
            new_clip = new_clip.with_mask()
        if new_clip.mask is not None:
            new_clip.mask = new_clip.mask.transform(make_filter())
        return new_clip
