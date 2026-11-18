"""Automatic alignment of audio recorded on separate devices.

Typical use case: a camera records a reference (scratch) audio track while
an external recorder captures the "real" audio. Both devices start and stop
independently, so the two tracks are shifted by a constant, unknown time
difference. This module estimates that difference by cross-correlation and
builds clips with the external audio re-attached on the reference timeline.

The main entry points are:

- ``estimate_audio_offset``: estimate the constant offset (in seconds) and
  a confidence score between a reference and a recorded ``AudioClip``.
- ``align_audio_clip``: shift a recorded ``AudioClip`` by an estimated
  offset so it plays in sync with the reference timeline.
- ``align_audio_to_video``: estimate the offset and return a copy of a
  ``VideoClip`` with the external audio attached, preserving the original
  frames and duration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from moviepy.audio.AudioClip import AudioClip, CompositeAudioClip


__all__ = [
    "AudioAlignmentError",
    "AudioOffsetEstimate",
    "AudioWindowEstimate",
    "estimate_audio_offset",
    "align_audio_clip",
    "align_audio_to_video",
]


#: Value of ``AudioOffsetEstimate.status`` when the estimation succeeded.
STATUS_OK = "ok"
#: The best offset sits at the edge of the searched range; the true offset
#: may lie outside ``[-max_offset, max_offset]``.
STATUS_BOUNDARY = "boundary"
#: One of the clips is (practically) silent, correlation is meaningless.
STATUS_SILENT = "silent"
#: Best normalized correlation is below ``min_confidence``.
STATUS_LOW_CORRELATION = "low_correlation"
#: Clips are too short (or ``max_offset`` too large) to correlate.
STATUS_INSUFFICIENT_SAMPLES = "insufficient_samples"
#: NaN or infinite samples were found while reading the clips.
STATUS_NON_FINITE = "non_finite"
#: Reading was cancelled through ``progress_callback``.
STATUS_CANCELLED = "cancelled"


class AudioAlignmentError(Exception):
    """Raised when an audio alignment cannot be performed or applied."""

    def __init__(self, message, estimate=None):
        super().__init__(message)
        self.estimate = estimate


@dataclass
class AudioWindowEstimate:
    """Offset estimated from a single analysis window.

    Attributes
    ----------

    start
      Start time (seconds) of the window on the reference timeline.

    offset
      Offset (seconds) estimated from this window, with the same sign
      convention as ``AudioOffsetEstimate.offset``.

    confidence
      Normalized correlation (0 to 1) at the best lag for this window.

    boundary
      True if the best lag of this window sits at the edge of the
      searched range.
    """

    start: float
    offset: float
    confidence: float
    boundary: bool = False


@dataclass
class AudioOffsetEstimate:
    """Result of :func:`estimate_audio_offset`.

    Attributes
    ----------

    offset
      Estimated constant time difference, in seconds, or ``None`` when no
      usable offset could be estimated. This is the time at which the
      recorded clip must start on the reference timeline to be in sync:
      ``recorded.with_start(offset)`` aligns it with the reference. A
      positive value means the recorder started *after* the reference
      device; a negative value means it started *before*.

    confidence
      Confidence score between 0 and 1 (normalized correlation of the
      agreeing analysis windows). 1 means a perfect match.

    status
      One of ``"ok"``, ``"boundary"``, ``"silent"``, ``"low_correlation"``,
      ``"insufficient_samples"``, ``"non_finite"`` or ``"cancelled"``.
      ``offset`` is only set for ``"ok"`` and ``"boundary"``.

    fps
      Sample rate at which the analysis was performed.

    windows
      Per-window estimates (:class:`AudioWindowEstimate`) used to compute
      the final offset.
    """

    offset: float | None
    confidence: float
    status: str
    fps: float | None = None
    windows: tuple = field(default_factory=tuple)

    @property
    def ok(self):
        """True if a usable offset was estimated."""
        return self.offset is not None

    def apply(self, recorded, duration=None):
        """Returns ``recorded`` shifted by this offset as a new AudioClip.

        Shortcut for ``align_audio_clip(recorded, self, duration)``.
        """
        return align_audio_clip(recorded, self, duration=duration)


def _failed_estimate(status, fps=None, confidence=0.0, windows=()):
    return AudioOffsetEstimate(
        offset=None, confidence=confidence, status=status, fps=fps, windows=windows
    )


def _to_mono(clip, tt, fps):
    """Reads ``clip`` at times ``tt`` and mixes down to a mono float array."""
    data = np.asarray(clip.to_soundarray(tt=tt, fps=fps), dtype=np.float64)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data


def _scan_energies(clip, fps, chunk_duration, progress_callback):
    """Computes per-chunk RMS energies of a clip, reading it chunk by chunk.

    Returns the array of energies, or ``None`` if the read was cancelled
    through ``progress_callback``. Raises ``_NonFiniteData`` if a chunk
    contains NaN or infinite values.
    """
    energies = []
    n_chunks = max(1, int(np.ceil(clip.duration / chunk_duration)))
    for i, chunk in enumerate(
        clip.iter_chunks(chunk_duration=chunk_duration, fps=fps, logger=None)
    ):
        chunk = np.asarray(chunk, dtype=np.float64)
        if not np.isfinite(chunk).all():
            raise _NonFiniteData()
        energies.append(float((chunk**2).mean()) if chunk.size else 0.0)
        if progress_callback is not None:
            if progress_callback("scan", i + 1, n_chunks) is False:
                return None
    return np.array(energies)


class _NonFiniteData(Exception):
    pass


def _best_lag_in_window(ref_seg, rec_seg, j_lo, j_hi, min_overlap):
    """Cross-correlates ``ref_seg`` against ``rec_seg`` for lags j_lo..j_hi.

    ``corr[j] = sum_i ref_seg[i] * rec_seg[j + i]`` (``j`` may be negative,
    with the segments then only partially overlapping). Each lag is
    normalized by the energy of the overlapping parts of both segments, so
    the result is a correlation coefficient. Lags whose overlap is shorter
    than ``min_overlap`` samples are ignored. Returns ``(j_best,
    confidence)`` or ``None`` if no lag has enough overlap.
    """
    length = len(ref_seg)
    n_fft = 1 << (len(rec_seg) + length - 2).bit_length()
    spectrum_rec = np.fft.rfft(rec_seg, n_fft)
    spectrum_ref = np.fft.rfft(ref_seg, n_fft)
    # Circular cross-correlation: c[j % n_fft] = sum_i rec_seg[j+i] * ref_seg[i]
    corr = np.fft.irfft(spectrum_rec * np.conj(spectrum_ref), n_fft)

    lags = np.arange(j_lo, j_hi + 1)
    values = corr[lags % n_fft]

    # Overlap between the two segments for each lag: ref_seg[i0:i1] is
    # compared with rec_seg[i0+j:i1+j].
    i0 = np.maximum(0, -lags)
    i1 = np.minimum(length, len(rec_seg) - lags)
    overlap = i1 - i0
    cumsum_ref = np.concatenate(([0.0], np.cumsum(ref_seg**2)))
    cumsum_rec = np.concatenate(([0.0], np.cumsum(rec_seg**2)))
    ref_norms = np.sqrt(cumsum_ref[i1] - cumsum_ref[i0])
    rec_norms = np.sqrt(cumsum_rec[i1 + lags] - cumsum_rec[i0 + lags])

    normalized = values / (ref_norms * rec_norms + 1e-20)
    normalized[overlap < min_overlap] = -np.inf
    best = int(np.argmax(normalized))
    if not np.isfinite(normalized[best]):
        return None
    return int(lags[best]), float(np.clip(normalized[best], -1.0, 1.0))


def estimate_audio_offset(
    reference,
    recorded,
    max_offset,
    fps=None,
    window_duration=30.0,
    max_windows=5,
    min_confidence=0.3,
    tolerance=0.02,
    silence_threshold=1e-4,
    chunk_duration=1.0,
    progress_callback=None,
):
    """Estimates the constant time offset between two audio recordings.

    Both clips are mixed down to mono and resampled to a common analysis
    rate, so they may differ in sample rate and channel count. The
    reference clip is scanned chunk by chunk (bounded memory) and the
    ``max_windows`` loudest windows of ``window_duration`` seconds are
    cross-correlated against the recorded clip; the final offset is the
    confidence-weighted average of the agreeing per-window offsets.

    Parameters
    ----------

    reference
      ``AudioClip`` taken as the timeline reference (e.g. the camera
      scratch audio). Must have a ``duration``.

    recorded
      ``AudioClip`` to align (e.g. the external recorder). Must have a
      ``duration``.

    max_offset
      Maximum absolute offset, in seconds, to search for. Must be
      positive and finite.

    fps
      Sample rate (Hz) used for the analysis. Defaults to the lowest of
      the two clips' ``fps`` attributes (44100 if none is set).

    window_duration
      Duration, in seconds, of each analysis window. Long audio is only
      analyzed through the selected windows. Should be comfortably longer
      than ``max_offset``: an offset can only be detected from a window
      if the two recordings still overlap inside it.

    max_windows
      Maximum number of windows (the loudest ones) to analyze.

    min_confidence
      Minimum normalized correlation for the result to be considered
      usable. Below it, the status is ``"low_correlation"``.

    tolerance
      Maximum spread, in seconds, between per-window offsets for them to
      be considered in agreement.

    silence_threshold
      RMS below which audio is considered silent.

    chunk_duration
      Duration, in seconds, of the chunks used when scanning the
      reference for loud windows.

    progress_callback
      Optional callable ``f(stage, done, total)`` called while reading
      audio (``stage`` is ``"scan"`` or ``"analyze"``). If it returns
      ``False``, the estimation aborts and the result has status
      ``"cancelled"``.

    Returns
    -------

    AudioOffsetEstimate
      The estimated offset and confidence. Check ``estimate.status`` (or
      ``estimate.ok``): ``offset`` is only set for statuses ``"ok"`` and
      ``"boundary"``.
    """
    if not np.isfinite(max_offset) or max_offset <= 0:
        raise ValueError("max_offset must be a positive, finite number of seconds")
    for name, clip in (("reference", reference), ("recorded", recorded)):
        if getattr(clip, "duration", None) is None:
            raise ValueError(f"{name} clip must have a duration to be aligned")
        if not np.isfinite(clip.duration) or clip.duration <= 0:
            raise ValueError(f"{name} clip has a non-finite or null duration")

    if fps is None:
        rates = [
            getattr(clip, "fps", None) for clip in (reference, recorded)
        ]
        rates = [rate for rate in rates if rate]
        fps = min(rates) if rates else 44100
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("analysis fps must be a positive, finite number")

    # Enough samples are needed for a meaningful correlation.
    min_samples = max(32, int(0.02 * fps))
    window_samples = int(
        min(window_duration, reference.duration, recorded.duration) * fps
    )
    if window_samples < min_samples:
        return _failed_estimate(STATUS_INSUFFICIENT_SAMPLES, fps=fps)

    # Scan the reference for the loudest analysis windows (bounded memory:
    # only per-chunk energies are kept, not the samples themselves).
    try:
        energies = _scan_energies(reference, fps, chunk_duration, progress_callback)
    except _NonFiniteData:
        return _failed_estimate(STATUS_NON_FINITE, fps=fps)
    if energies is None:
        return _failed_estimate(STATUS_CANCELLED, fps=fps)

    chunks_per_window = max(1, int(round(window_duration / chunk_duration)))
    n_windows = int(np.ceil(len(energies) / chunks_per_window))
    window_energies = np.array(
        [
            energies[i * chunks_per_window : (i + 1) * chunks_per_window].sum()
            for i in range(n_windows)
        ]
    )
    loud = np.argsort(window_energies)[::-1]
    selected = [
        int(i)
        for i in loud[:max_windows]
        if window_energies[i] > silence_threshold**2
    ]
    if not selected:
        return _failed_estimate(STATUS_SILENT, fps=fps)

    max_lag = int(round(max_offset * fps))
    windows = []
    n_silent_recorded = 0
    for w, window_index in enumerate(sorted(selected)):
        t0 = window_index * chunks_per_window * chunk_duration
        t1 = min(t0 + window_duration, reference.duration)
        tt_ref = np.arange(t0, t1, 1.0 / fps)
        if len(tt_ref) < min_samples:
            continue
        s0 = max(0.0, t0 - max_offset)
        s1 = min(recorded.duration, t1 + max_offset)
        tt_rec = np.arange(s0, s1, 1.0 / fps)

        try:
            ref_seg = _to_mono(reference, tt_ref, fps)
            rec_seg = _to_mono(recorded, tt_rec, fps)
        except (ValueError, IndexError):
            continue
        if not (np.isfinite(ref_seg).all() and np.isfinite(rec_seg).all()):
            return _failed_estimate(STATUS_NON_FINITE, fps=fps, windows=tuple(windows))

        length = len(ref_seg)
        if len(rec_seg) < min_samples:
            continue
        if np.sqrt((rec_seg**2).mean()) < silence_threshold:
            n_silent_recorded += 1

        # Lags j such that the offset stays within +-max_offset and the
        # compared segments overlap inside the extracted arrays. j is the
        # index, in rec_seg, at which ref_seg best matches:
        # offset = (t0 - s0) - j / fps.
        j_center = int(round((t0 - s0) * fps))
        j_lo = max(-(length - 1), j_center - max_lag)
        j_hi = min(len(rec_seg) - 1, j_center + max_lag)
        if j_hi < j_lo:
            continue

        min_overlap = max(min_samples, int(0.2 * length))
        best = _best_lag_in_window(ref_seg, rec_seg, j_lo, j_hi, min_overlap)
        if best is None:
            continue
        j_best, confidence = best
        offset = (t0 - s0) - j_best / fps
        boundary = j_best == j_lo or j_best == j_hi
        windows.append(
            AudioWindowEstimate(
                start=t0, offset=offset, confidence=confidence, boundary=boundary
            )
        )
        if progress_callback is not None:
            if progress_callback("analyze", w + 1, len(selected)) is False:
                return _failed_estimate(
                    STATUS_CANCELLED, fps=fps, windows=tuple(windows)
                )

    if not windows:
        return _failed_estimate(STATUS_INSUFFICIENT_SAMPLES, fps=fps)
    if n_silent_recorded == len(windows):
        return _failed_estimate(STATUS_SILENT, fps=fps, windows=tuple(windows))

    # Keep the windows whose offsets agree with the (robust) median offset.
    offsets = np.array([w.offset for w in windows])
    median = np.median(offsets)
    inliers = [w for w in windows if abs(w.offset - median) <= tolerance]
    weights = np.array([max(w.confidence, 1e-6) for w in inliers])
    offset = float(np.average([w.offset for w in inliers], weights=weights))
    confidence = float(np.average([w.confidence for w in inliers], weights=weights))
    confidence = float(np.clip(confidence, 0.0, 1.0))

    if confidence < min_confidence:
        return _failed_estimate(
            STATUS_LOW_CORRELATION,
            fps=fps,
            confidence=confidence,
            windows=tuple(windows),
        )
    if abs(offset) >= max_offset - 1.5 / fps:
        status = STATUS_BOUNDARY
    else:
        status = STATUS_OK
    return AudioOffsetEstimate(
        offset=offset,
        confidence=confidence,
        status=status,
        fps=fps,
        windows=tuple(windows),
    )


def align_audio_clip(recorded, estimate, duration=None):
    """Returns a new AudioClip of ``recorded`` shifted by ``estimate.offset``.

    The returned clip plays the recorded audio on the reference timeline:
    its content at time ``t`` is ``recorded`` at time ``t - offset``.
    Times outside the recorded clip are silent.

    Parameters
    ----------

    recorded
      The ``AudioClip`` to shift.

    estimate
      An ``AudioOffsetEstimate`` with a usable offset (status ``"ok"`` or
      ``"boundary"``), as returned by ``estimate_audio_offset``.

    duration
      Optional duration for the returned clip (e.g. the reference video
      duration). If ``None``, the duration is ``offset + recorded
      duration``.

    Raises
    ------

    AudioAlignmentError
      If ``estimate`` has no usable offset.
    """
    if estimate is None or estimate.offset is None:
        raise AudioAlignmentError(
            f"Cannot align audio: offset estimation failed "
            f"(status={estimate.status!r}).",
            estimate=estimate,
        )
    aligned = CompositeAudioClip([recorded.with_start(estimate.offset)])
    if duration is not None:
        aligned = aligned.with_duration(duration)
    return aligned


def align_audio_to_video(
    video,
    recorded,
    max_offset=None,
    estimate=None,
    reference=None,
    **estimate_kwargs,
):
    """Returns a copy of ``video`` with ``recorded`` audio attached, in sync.

    The offset between the reference audio (``reference``, or the video's
    own audio track) and the externally ``recorded`` audio is estimated
    with :func:`estimate_audio_offset`, and the recorded audio is shifted
    accordingly. The returned clip has exactly the same frames and
    duration as ``video``; only its audio track changes. The estimate is
    available as the ``audio_alignment`` attribute of the returned clip.

    Parameters
    ----------

    video
      A ``VideoClip``. Its frames and duration are preserved.

    recorded
      The externally recorded ``AudioClip`` to attach.

    max_offset
      Maximum absolute offset, in seconds, to search for. Required unless
      ``estimate`` is provided.

    estimate
      A pre-computed ``AudioOffsetEstimate``. If given, ``max_offset``,
      ``reference`` and the other estimation parameters are ignored.

    reference
      ``AudioClip`` to use as the alignment reference. Defaults to
      ``video.audio``.

    estimate_kwargs
      Additional keyword arguments for :func:`estimate_audio_offset`
      (``fps``, ``window_duration``, ``max_windows``, ``min_confidence``,
      etc.).

    Raises
    ------

    AudioAlignmentError
      If the offset could not be estimated (silence, low correlation,
      insufficient samples, non-finite data or cancelled read).
    """
    if estimate is None:
        if max_offset is None:
            raise ValueError("max_offset is required when estimate is not provided")
        if reference is None:
            reference = video.audio
        if reference is None:
            raise ValueError(
                "No reference audio: pass reference= or use a video with audio"
            )
        estimate = estimate_audio_offset(
            reference, recorded, max_offset, **estimate_kwargs
        )
    aligned_audio = align_audio_clip(recorded, estimate, duration=video.duration)
    result = video.with_audio(aligned_audio)
    result.audio_alignment = estimate
    return result
