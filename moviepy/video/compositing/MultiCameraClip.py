"""Multi-camera grid composition.

This module provides :class:`MultiCameraClip`, a :class:`CompositeVideoClip`
specialization that arranges several camera feeds (which may have different
resolutions, aspect ratios and start times) into a fixed-size, fixed-duration
synchronized grid (for instance a stable 2x2 interview layout).

Camera inputs are accepted either as ``(clip, offset, (row, col))`` tuples or
as dictionaries with the ``clip``/``offset``/``position`` keys.
"""

import numbers

from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.fx.MultiplyVolume import MultiplyVolume
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.fx.Crop import Crop
from moviepy.video.fx.Resize import Resize


_REQUIRED_CONFIG_KEYS = ("clip", "offset", "position")

_FIT_MODES = ("contain", "cover")
_AUDIO_MODES = ("host", "mix")


class MultiCameraClip(CompositeVideoClip):
    """A synchronized grid of camera feeds with different starts and sizes.

    Each camera is placed in one cell of a ``rows x cols`` grid. Feeds are
    time-aligned through a per-camera offset, fitted to the cell size either by
    proportional letterboxing (``"contain"``) or centered cropping
    (``"cover"``), and the whole composition always renders at the exact
    requested ``size`` and ``duration``. Regions where a camera has no frame
    (before it starts, after it ends, or the letterbox borders) are filled with
    ``bg_color``.

    Parameters
    ----------

    cameras
      The camera feeds. Each entry is either a tuple
      ``(clip, offset, (row, col))`` or a dictionary
      ``{"clip": clip, "offset": offset, "position": (row, col)}``, where:

      - ``clip`` is any ``VideoClip`` (it must have a ``size``);
      - ``offset`` is the time, in seconds, at which the clip's first frame is
        shown in the output. Only non-negative offsets are accepted;
      - ``(row, col)`` is the zero-based grid position, with ``(0, 0)`` at the
        top-left. Two cameras cannot share the same position.

    size
      The output size ``(width, height)`` in pixels. It must be divisible by
      respectively the number of columns and rows, so that every grid cell has
      the same integer size.

    duration
      The output duration in seconds. Always finite and strictly positive. The
      resulting clip always has exactly this duration, regardless of the
      (possibly infinite) durations of the input clips.

    grid_shape
      The grid shape ``(rows, cols)``. Defaults to ``(2, 2)``.

    fit
      How a source frame is fitted into its cell:

      - ``"contain"`` (default): scale while preserving the aspect ratio so the
        whole frame is visible, and fill the remaining borders with
        ``bg_color`` (letterboxing / pillarboxing);
      - ``"cover"``: scale while preserving the aspect ratio so the whole cell
        is covered, then center-crop the overflowing parts.

    bg_color
      RGB color used for the empty cells, the letterbox borders and every
      region without a frame. Defaults to black ``(0, 0, 0)``.

    audio_mode
      How the output soundtrack is built:

      - ``"host"`` (default): only the host camera's audio is kept;
      - ``"mix"``: the audios of every camera that has one are summed.

    host_index
      Index, in the ``cameras`` list, of the host camera used when
      ``audio_mode="host"``. Defaults to ``0``. The host clip must have an
      audio track.

    audio_weights
      Optional mapping ``{camera_index: weight}`` used when
      ``audio_mode="mix"``. Cameras missing from the mapping get a weight of
      ``1.0``. Weights are applied as volume multipliers; the tracks are then
      summed (use weights below ``1.0`` to avoid clipping).

    Examples
    --------

    .. code:: python

        from moviepy import VideoFileClip, MultiCameraClip

        cam_a = VideoFileClip("cam_a.mp4")
        cam_b = VideoFileClip("cam_b.mp4")
        cam_c = VideoFileClip("cam_c.mp4")
        cam_d = VideoFileClip("cam_d.mp4")

        grid = MultiCameraClip(
            cameras=[
                (cam_a, 0.0, (0, 0)),
                (cam_b, 0.5, (0, 1)),  # cam_b starts 0.5s later
                (cam_c, 1.0, (1, 0)),
                (cam_d, 0.0, (1, 1)),
            ],
            size=(1280, 720),
            duration=10,
            fit="contain",
            audio_mode="host",
            host_index=0,
        )
        grid.write_videofile("interview_grid.mp4", fps=25)
        grid.close()

    Rules for invalid inputs
    ------------------------

    All of the following raise a ``ValueError``:

    - two cameras share the same grid position;
    - a camera offset is negative, or greater than or equal to the output
      duration;
    - the output ``duration`` is not a finite positive number — input clips
      with an infinite duration (``duration is None``) are fine, they are
      automatically subclipped to the finite output duration;
    - the requested ``size`` is not a tuple of positive integers divisible by
      the grid dimensions;
    - a grid position is outside ``grid_shape``;
    - ``audio_mode="host"`` but the host clip has no audio track. Clips
      without audio are silently skipped in ``"mix"`` mode; if no clip has any
      audio, the result simply has no soundtrack.
    """

    def __init__(
        self,
        cameras,
        size,
        duration,
        grid_shape=(2, 2),
        fit="contain",
        bg_color=(0, 0, 0),
        audio_mode="host",
        host_index=0,
        audio_weights=None,
    ):
        configs = [
            self._normalize_config(camera, index)
            for index, camera in enumerate(cameras)
        ]

        # --- Structural validation ---------------------------------------
        if fit not in _FIT_MODES:
            raise ValueError("fit must be one of %s, got %r" % (_FIT_MODES, fit))
        if audio_mode not in _AUDIO_MODES:
            raise ValueError(
                "audio_mode must be one of %s, got %r" % (_AUDIO_MODES, audio_mode)
            )

        rows, cols = grid_shape
        if (
            not isinstance(rows, numbers.Integral)
            or not isinstance(cols, numbers.Integral)
            or rows <= 0
            or cols <= 0
        ):
            raise ValueError(
                "grid_shape must be two positive integers (rows, cols), got %r"
                % (grid_shape,)
            )

        if (
            not isinstance(size, (tuple, list))
            or len(size) != 2
            or not all(isinstance(v, numbers.Integral) and v > 0 for v in size)
        ):
            raise ValueError(
                "size must be a (width, height) tuple of positive integers, got %r"
                % (size,)
            )
        width, height = size
        cell_w, cell_h = width // cols, height // rows
        if cell_w * cols != width or cell_h * rows != height:
            raise ValueError(
                "size %r is not divisible by grid_shape %r: cell size would be "
                "non-integer (%d columns, %d rows)." % (size, grid_shape, cols, rows)
            )

        if not isinstance(duration, numbers.Real) or duration <= 0:
            raise ValueError(
                "duration must be a finite positive number of seconds, got %r"
                % (duration,)
            )

        if not isinstance(host_index, numbers.Integral) or not (
            0 <= host_index < len(configs)
        ):
            raise ValueError(
                "host_index must be an index in [0, %d), got %r"
                % (len(configs), host_index)
            )

        if not isinstance(bg_color, (tuple, list)) or len(bg_color) != 3:
            raise ValueError(
                "bg_color must be an RGB (r, g, b) tuple, got %r" % (bg_color,)
            )

        seen_positions = set()
        for index, config in enumerate(configs):
            clip, offset, position = (
                config["clip"],
                config["offset"],
                config["position"],
            )
            if not isinstance(offset, numbers.Real) or offset < 0:
                raise ValueError(
                    "Camera %d has an invalid offset %r: offsets must be "
                    "non-negative numbers of seconds." % (index, offset)
                )
            if offset >= duration:
                raise ValueError(
                    "Camera %d offset (%.3fs) is greater than or equal to the "
                    "output duration (%.3fs): it would never be visible."
                    % (index, offset, duration)
                )
            if (
                not isinstance(position, tuple)
                or len(position) != 2
                or not all(isinstance(v, numbers.Integral) for v in position)
            ):
                raise ValueError(
                    "Camera %d has an invalid position %r: expected an "
                    "(row, col) tuple of integers." % (index, position)
                )
            row, col = position
            if not (0 <= row < rows and 0 <= col < cols):
                raise ValueError(
                    "Camera %d position %r is outside the %r grid."
                    % (index, position, grid_shape)
                )
            if position in seen_positions:
                raise ValueError(
                    "Duplicate grid position %r: cameras %d and another camera "
                    "cannot share the same cell." % (position, index)
                )
            seen_positions.add(position)
            if getattr(clip, "size", None) is None:
                raise ValueError(
                    "Camera %d clip has no size attribute: only VideoClip "
                    "instances can be composited." % index
                )

        audio_weights = audio_weights or {}
        for index, weight in audio_weights.items():
            if not isinstance(index, numbers.Integral) or not (
                0 <= index < len(configs)
            ):
                raise ValueError(
                    "audio_weights key %r is not a valid camera index in [0, %d)."
                    % (index, len(configs))
                )
            if not isinstance(weight, numbers.Real) or weight < 0:
                raise ValueError(
                    "audio_weights[%d] must be a non-negative number, got %r"
                    % (index, weight)
                )
        self.cameras = configs
        self.grid_shape = grid_shape
        self.cell_size = (cell_w, cell_h)
        self.fit = fit
        self.host_index = host_index
        self.audio_weights = dict(audio_weights)

        # --- Build one self-contained, audio-free composite per cell ------
        cells = []
        for config in configs:
            clip = config["clip"]
            offset = config["offset"]
            row, col = config["position"]

            # Bounding the source by the finite output duration also makes
            # infinite clips safe to render.
            source = self._limit_clip(clip, duration).without_audio()
            cell = self._build_cell(source, (cell_w, cell_h), fit, bg_color)
            cell = (
                cell.with_duration(duration)
                .with_start(offset)
                .with_position((col * cell_w, row * cell_h))
            )
            cells.append(cell)

        # CompositeVideoClip computes the duration from the cell ends. Every
        # cell starts at t=0 at the latest, so the result spans [0, duration].
        super().__init__(cells, size=size, bg_color=bg_color)
        self.duration = duration
        self.end = duration

        # --- Build the soundtrack separately for precise control ----------
        self.audio = self._build_audio(
            configs, duration, audio_mode, host_index, audio_weights
        )
        if self.audio is not None:
            self.audio = self.audio.with_duration(duration)

    @staticmethod
    def _normalize_config(camera, index):
        """Turn a tuple/dictionary camera spec into a normalized dict."""
        if isinstance(camera, dict):
            missing = [key for key in _REQUIRED_CONFIG_KEYS if key not in camera]
            if missing:
                raise ValueError(
                    "Camera %d dictionary is missing required key(s): %s"
                    % (index, ", ".join(missing))
                )
            return {
                "clip": camera["clip"],
                "offset": camera["offset"],
                "position": camera["position"],
            }
        if isinstance(camera, (tuple, list)) and len(camera) == 3:
            clip, offset, position = camera
            return {"clip": clip, "offset": offset, "position": position}
        raise ValueError(
            "Camera %d must be a (clip, offset, (row, col)) tuple or a "
            "{'clip', 'offset', 'position'} dictionary, got %r" % (index, camera)
        )

    @staticmethod
    def _limit_clip(clip, duration):
        """Return ``clip`` restricted to at most ``duration`` seconds.

        Finite clips shorter than ``duration`` are returned untouched: they
        simply stop playing and their cell falls back to the background.
        Infinite clips (``duration is None``) are subclipped so the finite
        composition can always be rendered.
        """
        if clip.duration is None:
            return clip.subclipped(0, duration)
        if clip.duration > duration:
            return clip.subclipped(0, duration)
        return clip

    @staticmethod
    def _build_cell(source, cell_size, fit, bg_color):
        """Fit ``source`` into a clip of exactly ``cell_size`` pixels."""
        cell_w, cell_h = cell_size
        src_w, src_h = source.size

        if fit == "contain":
            scale = min(cell_w / src_w, cell_h / src_h)
            new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
            fitted = source.with_effects([Resize((new_w, new_h))])
            return CompositeVideoClip(
                [fitted.with_position("center")],
                size=cell_size,
                bg_color=bg_color,
            )

        # "cover": fill the cell entirely, then center-crop the overflow.
        scale = max(cell_w / src_w, cell_h / src_h)
        new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
        resized = source.with_effects([Resize((new_w, new_h))])
        x1 = max(0, (new_w - cell_w) // 2)
        y1 = max(0, (new_h - cell_h) // 2)
        cropped = resized.with_effects(
            [Crop(x1=x1, y1=y1, width=cell_w, height=cell_h)]
        )
        # Crop may leave a frame a pixel short because of rounding; embed it
        # in a cell-sized composite to guarantee a constant cell size.
        return CompositeVideoClip(
            [cropped.with_position("center")],
            size=cell_size,
            bg_color=bg_color,
        )

    @staticmethod
    def _build_audio(configs, duration, audio_mode, host_index, audio_weights):
        """Build the aligned output soundtrack, or ``None`` if there is none."""
        if audio_mode == "host":
            host = configs[host_index]["clip"]
            if host.audio is None:
                raise ValueError(
                    "Host camera (index %d) has no audio track. Pick another "
                    "host_index or use audio_mode='mix'." % host_index
                )
            track = MultiCameraClip._limit_clip(host.audio, duration)
            return track.with_start(configs[host_index]["offset"])

        tracks = []
        for index, config in enumerate(configs):
            audio = config["clip"].audio
            if audio is None:
                # Missing tracks are simply absent from the mix.
                continue
            weight = audio_weights.get(index, 1.0)
            track = MultiCameraClip._limit_clip(audio, duration)
            if weight != 1.0:
                track = track.with_effects([MultiplyVolume(weight)])
            tracks.append(track.with_start(config["offset"]))

        if not tracks:
            return None
        return CompositeAudioClip(tracks)

    def close(self):
        """Release the resources created by this composition.

        Source clips remain owned by the caller and are deliberately not
        closed: the same input feed may be shared between compositions. Only
        the background clip, the composite audio and the nested cell clips
        created internally are released.
        """
        for cell in getattr(self, "clips", []):
            try:
                cell.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
        super().close()
