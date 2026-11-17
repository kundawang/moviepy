"""Animate the opacity of a clip from time-based keyframes."""

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Tuple, Union

from moviepy.Clip import Clip
from moviepy.Effect import Effect

KeyframesType = Union[Mapping[float, float], Sequence[Tuple[float, float]]]


def _normalise_keyframes(
    keyframes: KeyframesType,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Validate ``keyframes`` and return two sorted tuples ``(times, opacities)``.

    Keyframes are given in the clip's local time (in seconds). They may be
    supplied as a mapping ``{time: opacity}`` or as a sequence of
    ``(time, opacity)`` pairs, in any order.

    Raises
    ------
    ValueError
        If the keyframes are empty, malformed, contain a non-finite number,
        a negative time, a duplicated time, or an opacity outside ``[0, 1]``.
    """
    if isinstance(keyframes, Mapping):
        entries = list(keyframes.items())
    elif isinstance(keyframes, (str, bytes)):
        raise ValueError(
            "Opacity keyframes must be a mapping or a sequence of (time, opacity) "
            "pairs, not a string"
        )
    else:
        try:
            entries = list(keyframes)
        except TypeError:
            raise ValueError(
                "Opacity keyframes must be a mapping or a sequence of "
                "(time, opacity) pairs"
            ) from None

    if not entries:
        raise ValueError("Opacity keyframes must contain at least one keyframe")

    normalised = []
    for entry in entries:
        if isinstance(entry, (str, bytes)):
            raise ValueError(
                f"Invalid opacity keyframe {entry!r}: expected a (time, opacity) pair"
            )

        try:
            time, opacity = entry
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid opacity keyframe {entry!r}: expected a (time, opacity) pair"
            ) from None

        if isinstance(time, bool) or not isinstance(time, Real):
            raise ValueError(
                f"Invalid opacity keyframe time {time!r}: must be a real number"
            )
        if isinstance(opacity, bool) or not isinstance(opacity, Real):
            raise ValueError(
                f"Invalid opacity keyframe value {opacity!r}: must be a real number"
            )

        time, opacity = float(time), float(opacity)

        if not isfinite(time):
            raise ValueError(
                f"Invalid opacity keyframe time {time!r}: must be a finite number"
            )
        if not isfinite(opacity):
            raise ValueError(
                f"Invalid opacity keyframe value {opacity!r}: must be a finite number"
            )
        if time < 0:
            raise ValueError(
                f"Invalid opacity keyframe time {time!r}: must be greater or equal to 0"
            )
        if not 0 <= opacity <= 1:
            raise ValueError(
                f"Invalid opacity keyframe value {opacity!r}: must be between 0 and 1"
            )

        normalised.append((time, opacity))

    normalised.sort(key=lambda keyframe: keyframe[0])

    for (previous_time, _), (time, _) in zip(normalised, normalised[1:]):
        if time == previous_time:
            raise ValueError(
                f"Duplicated opacity keyframe time {time!r}: each keyframe time "
                "must appear only once"
            )

    times, opacities = zip(*normalised)
    return times, opacities


def _interpolate_opacity(
    time: float, times: Tuple[float, ...], opacities: Tuple[float, ...]
) -> float:
    """Return the linearly interpolated opacity at ``time``.

    Times before the first keyframe hold the first opacity and times after the
    last keyframe hold the last opacity, so the effect never extrapolates.
    Keyframe times are strictly increasing, as guaranteed by
    ``_normalise_keyframes``.
    """
    if time <= times[0]:
        return opacities[0]
    if time >= times[-1]:
        return opacities[-1]

    index = bisect_right(times, time)
    previous_time, next_time = times[index - 1], times[index]
    previous_opacity, next_opacity = opacities[index - 1], opacities[index]

    progress = (time - previous_time) / (next_time - previous_time)
    return previous_opacity + progress * (next_opacity - previous_opacity)


@dataclass
class OpacityKeyframes(Effect):
    """Animate a clip's opacity from keyframes expressed in clip-local time.

    Keyframes are ``(time, opacity)`` pairs, with times in seconds relative to
    the beginning of the clip and opacities between 0 (fully transparent) and
    1 (fully opaque). They may be given as a mapping ``{time: opacity}`` or as
    a sequence of pairs, in any order. Values are linearly interpolated at any
    requested frame time, including times falling between two video frames.
    Before the first keyframe the first opacity is used, and after the last
    keyframe the last opacity is used.

    The interpolated opacity is multiplied with the clip's existing mask, so
    a clip that already has partial transparency (for example an image with an
    alpha channel) keeps it. Clips without a mask get a fully opaque mask
    first. The clip's audio is left untouched.

    Because times are clip-local, the effect follows the clip through
    ``subclipped``, ``with_start`` and ``with_speed_scaled``: when the clip is
    placed in a ``CompositeVideoClip``, opacity is evaluated at the same
    visible time as the frames. Apply the effect before a speed change to keep
    keyframes in the original timeline, or after it to express them in the new
    timeline.

    The clip must have a duration, because keyframe times are checked against
    it: every keyframe time must fall within ``[0, duration]``. Like other
    time-based effects (``FadeOut``, ``CrossFadeIn``, ...), applying the effect
    to a clip whose ``duration`` is ``None`` raises a ``ValueError``.

    Parameters
    ----------
    keyframes
        Mapping ``{time: opacity}`` or sequence of ``(time, opacity)`` pairs.
        Times must be finite, non-negative and unique. Opacities must be
        finite and between 0 and 1.

    Raises
    ------
    ValueError
        If keyframes are empty, malformed, duplicated, non-finite, have
        negative times, fall after the clip's duration, or contain opacity
        values outside ``[0, 1]``. Also raised if the clip has no duration.

    Examples
    --------
    Fade a caption in, hold it, then fade it out, while keeping a png's own
    transparency:

    .. code:: python

        caption = (
            ImageClip("caption.png")
            .with_duration(4)
            .with_effects(
                [vfx.OpacityKeyframes([(0, 0), (0.5, 1), (3, 1), (3.5, 0)])]
            )
        )
        video = CompositeVideoClip([background, caption.with_start(2)])
    """

    keyframes: KeyframesType

    def apply(self, clip: Clip) -> Clip:
        """Apply the opacity keyframes to the clip's mask."""
        times, opacities = _normalise_keyframes(self.keyframes)

        if clip.duration is None:
            raise ValueError("Attribute 'duration' not set")

        last_time = times[-1]
        # Tolerate the usual floating point error on a keyframe placed
        # exactly at the clip's end.
        if last_time - clip.duration > 1e-9:
            raise ValueError(
                f"Opacity keyframe time {last_time!r} is outside the clip's "
                f"duration ({clip.duration!r})"
            )

        if getattr(clip, "is_mask", False):
            # A mask clip is itself an alpha channel, so there is no mask of
            # its own to multiply: transform the mask frames directly.
            return clip.transform(
                lambda get_frame, time: get_frame(time)
                * _interpolate_opacity(time, times, opacities),
                keep_duration=True,
            )

        if clip.mask is None:
            clip = clip.with_mask()

        if clip.mask.duration is None:
            # Give the freshly created mask the clip's duration so it can be
            # composed and subclipped like the clip it belongs to.
            clip.mask = clip.mask.with_duration(clip.duration)

        clip.mask = clip.mask.transform(
            lambda get_frame, time: get_frame(time)
            * _interpolate_opacity(time, times, opacities)
        )
        return clip
