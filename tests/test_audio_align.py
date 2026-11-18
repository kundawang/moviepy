"""Tests for automatic audio alignment (moviepy.audio.tools.align)."""

import numpy as np
import pytest

from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.audio.tools.align import (
    AudioAlignmentError,
    align_audio_clip,
    align_audio_to_video,
    estimate_audio_offset,
)
from moviepy.video.VideoClip import ColorClip


FPS = 8000
REF_START = 10.0  # start of the reference (camera) track in "world" time
REF_DURATION = 10.0
REC_DURATION = 13.0


def world_signal(seed=0):
    """A 30 s 'world' signal: broadband noise bursts separated by silence."""
    rng = np.random.default_rng(seed)
    signal = np.zeros(30 * FPS)
    for start in [2.0, 6.0, 11.0, 16.0, 21.0, 26.0]:
        i0, i1 = int(start * FPS), int((start + 2.0) * FPS)
        signal[i0:i1] = 0.5 * rng.standard_normal(i1 - i0)
    return signal


def make_clips(offset, noise=0.05, fps_rec=None, stereo=False, seed=0):
    """Builds (reference, recorded) AudioArrayClips with a known offset.

    ``offset`` is where the recorded clip starts on the reference
    timeline: recorded[s] == reference[s + offset] (plus noise).
    """
    signal = world_signal(seed)
    i0 = int(REF_START * FPS)
    ref = signal[i0 : i0 + int(REF_DURATION * FPS)]
    j0 = int((REF_START + offset) * FPS)
    rec = signal[j0 : j0 + int(REC_DURATION * FPS)].copy()
    if fps_rec is not None and fps_rec != FPS:
        rec = np.repeat(rec, fps_rec // FPS)
    rng = np.random.default_rng(seed + 1)
    rec += noise * rng.standard_normal(len(rec))
    if stereo:
        rec = np.column_stack([rec, rec + noise * rng.standard_normal(len(rec))])
        rec_array = rec
    else:
        rec_array = rec.reshape(-1, 1)
    reference_clip = AudioArrayClip(ref.reshape(-1, 1), fps=FPS)
    recorded_clip = AudioArrayClip(rec_array, fps=fps_rec or FPS)
    return reference_clip, recorded_clip


def test_estimate_negative_offset():
    # recorder started 1.3 s before the camera
    reference, recorded = make_clips(offset=-1.3)
    estimate = estimate_audio_offset(reference, recorded, max_offset=2.0)
    assert estimate.ok
    assert estimate.status == "ok"
    assert estimate.offset == pytest.approx(-1.3, abs=2.0 / FPS)
    assert estimate.confidence > 0.9


def test_estimate_positive_offset():
    # recorder started 0.8 s after the camera
    reference, recorded = make_clips(offset=0.8)
    estimate = estimate_audio_offset(reference, recorded, max_offset=2.0)
    assert estimate.ok
    assert estimate.offset == pytest.approx(0.8, abs=2.0 / FPS)
    assert estimate.confidence > 0.9


def test_estimate_with_background_noise():
    reference, recorded = make_clips(offset=-1.3, noise=0.5)
    estimate = estimate_audio_offset(reference, recorded, max_offset=2.0)
    assert estimate.ok
    assert estimate.offset == pytest.approx(-1.3, abs=2.0 / FPS)
    # heavy noise must lower the confidence
    assert 0.1 < estimate.confidence < 0.9


def test_estimate_different_sample_rates():
    reference, recorded = make_clips(offset=-1.3, fps_rec=2 * FPS)
    estimate = estimate_audio_offset(reference, recorded, max_offset=2.0)
    assert estimate.ok
    assert estimate.offset == pytest.approx(-1.3, abs=2.0 / FPS)


def test_estimate_different_channel_counts():
    reference, recorded = make_clips(offset=-1.3, stereo=True)
    estimate = estimate_audio_offset(reference, recorded, max_offset=2.0)
    assert estimate.ok
    assert estimate.offset == pytest.approx(-1.3, abs=2.0 / FPS)


def test_estimate_windowed_analysis():
    # long audio is analyzed through a bounded number of selected windows
    reference, recorded = make_clips(offset=-1.3)
    estimate = estimate_audio_offset(
        reference, recorded, max_offset=2.0, window_duration=2.0, max_windows=3
    )
    assert estimate.ok
    assert estimate.offset == pytest.approx(-1.3, abs=2.0 / FPS)
    assert 1 <= len(estimate.windows) <= 3
    # all windows agree on the same constant offset
    for window in estimate.windows:
        assert window.offset == pytest.approx(-1.3, abs=2.0 / FPS)


def test_estimate_silent_reference():
    reference, recorded = make_clips(offset=-1.3)
    silent = AudioArrayClip(np.zeros((FPS, 1)), fps=FPS).with_duration(REF_DURATION)
    estimate = estimate_audio_offset(silent, recorded, max_offset=2.0)
    assert estimate.status == "silent"
    assert estimate.offset is None
    assert estimate.confidence == 0.0


def test_estimate_silent_recording():
    reference, recorded = make_clips(offset=-1.3)
    silent = AudioArrayClip(np.zeros((int(REC_DURATION * FPS), 1)), fps=FPS)
    estimate = estimate_audio_offset(reference, silent, max_offset=2.0)
    assert estimate.status == "silent"
    assert estimate.offset is None


def test_estimate_low_correlation():
    reference, _ = make_clips(offset=-1.3)
    rng = np.random.default_rng(42)
    unrelated = AudioArrayClip(
        (0.5 * rng.standard_normal((int(REC_DURATION * FPS), 1))), fps=FPS
    )
    estimate = estimate_audio_offset(reference, unrelated, max_offset=2.0)
    assert estimate.status == "low_correlation"
    assert estimate.offset is None
    assert estimate.confidence < 0.3


def test_estimate_insufficient_samples():
    tiny_ref = AudioArrayClip(np.zeros((80, 1)), fps=FPS)
    tiny_rec = AudioArrayClip(np.zeros((80, 1)), fps=FPS)
    estimate = estimate_audio_offset(tiny_ref, tiny_rec, max_offset=0.001)
    assert estimate.status == "insufficient_samples"
    assert estimate.offset is None


def test_estimate_boundary_offset():
    # offset exactly at the edge of the searched range
    reference, recorded = make_clips(offset=-2.0)
    estimate = estimate_audio_offset(reference, recorded, max_offset=2.0)
    assert estimate.status == "boundary"
    assert estimate.offset == pytest.approx(-2.0, abs=2.0 / FPS)

    reference, recorded = make_clips(offset=2.0)
    estimate = estimate_audio_offset(reference, recorded, max_offset=2.0)
    assert estimate.status == "boundary"
    assert estimate.offset == pytest.approx(2.0, abs=2.0 / FPS)


def test_estimate_non_finite_data():
    reference, recorded = make_clips(offset=-1.3)
    recorded.array[1000] = np.nan
    estimate = estimate_audio_offset(reference, recorded, max_offset=2.0)
    assert estimate.status == "non_finite"
    assert estimate.offset is None

    reference, recorded = make_clips(offset=-1.3)
    reference.array[5000] = np.inf
    estimate = estimate_audio_offset(reference, recorded, max_offset=2.0)
    assert estimate.status == "non_finite"
    assert estimate.offset is None


def test_estimate_cancelled_read():
    reference, recorded = make_clips(offset=-1.3)

    def cancel_on_scan(stage, done, total):
        return False

    estimate = estimate_audio_offset(
        reference, recorded, max_offset=2.0, progress_callback=cancel_on_scan
    )
    assert estimate.status == "cancelled"
    assert estimate.offset is None

    def cancel_on_analyze(stage, done, total):
        return stage != "analyze"

    estimate = estimate_audio_offset(
        reference, recorded, max_offset=2.0, progress_callback=cancel_on_analyze
    )
    assert estimate.status == "cancelled"
    assert estimate.offset is None


def test_estimate_invalid_arguments():
    reference, recorded = make_clips(offset=-1.3)
    with pytest.raises(ValueError):
        estimate_audio_offset(reference, recorded, max_offset=0)
    with pytest.raises(ValueError):
        estimate_audio_offset(reference, recorded, max_offset=np.inf)
    no_duration = AudioArrayClip(reference.array, fps=FPS)
    no_duration.duration = None
    with pytest.raises(ValueError):
        estimate_audio_offset(no_duration, recorded, max_offset=2.0)


def test_align_audio_clip_shifts_waveform():
    offset = -1.3
    reference, recorded = make_clips(offset=offset)
    estimate = estimate_audio_offset(reference, recorded, max_offset=2.0)
    aligned = align_audio_clip(recorded, estimate, duration=REF_DURATION)

    # aligned audio at time t is the recorded audio at t - offset
    tt = np.arange(2.0, 4.0, 1.0 / FPS)
    got = aligned.to_soundarray(tt=tt, fps=FPS)
    expected = recorded.to_soundarray(tt=tt - estimate.offset, fps=FPS)
    assert np.allclose(got, expected, atol=1e-6)

    # and it now matches the reference where they overlap
    ref_wave = reference.to_soundarray(tt=tt, fps=FPS)
    got_mono = got.mean(axis=1) if got.ndim == 2 else got
    ref_mono = ref_wave.mean(axis=1) if ref_wave.ndim == 2 else ref_wave
    correlation = np.corrcoef(got_mono, ref_mono)[0, 1]
    assert correlation > 0.9


def test_align_audio_clip_positive_offset_leaves_silence():
    offset = 0.8
    reference, recorded = make_clips(offset=offset)
    estimate = estimate_audio_offset(reference, recorded, max_offset=2.0)
    aligned = align_audio_clip(recorded, estimate, duration=REF_DURATION)
    # nothing is playable before the (positive) offset
    assert np.allclose(aligned.get_frame(0.1), 0.0)
    assert aligned.duration == REF_DURATION


def test_align_audio_clip_failed_estimate_raises():
    reference, recorded = make_clips(offset=-1.3)
    silent = AudioArrayClip(np.zeros((int(REC_DURATION * FPS), 1)), fps=FPS)
    estimate = estimate_audio_offset(reference, silent, max_offset=2.0)
    with pytest.raises(AudioAlignmentError):
        align_audio_clip(recorded, estimate)


def test_estimate_apply_method():
    reference, recorded = make_clips(offset=-1.3)
    estimate = estimate_audio_offset(reference, recorded, max_offset=2.0)
    aligned = estimate.apply(recorded, duration=REF_DURATION)
    assert aligned.duration == REF_DURATION
    tt = np.arange(2.0, 3.0, 1.0 / FPS)
    assert np.allclose(
        aligned.to_soundarray(tt=tt, fps=FPS),
        recorded.to_soundarray(tt=tt - estimate.offset, fps=FPS),
        atol=1e-6,
    )


def test_align_audio_to_video_preserves_frames_and_duration():
    offset = -1.3
    reference, recorded = make_clips(offset=offset)
    video = ColorClip((32, 24), color=(10, 200, 30), duration=REF_DURATION)
    video = video.with_audio(reference)

    aligned_video = align_audio_to_video(video, recorded, max_offset=2.0)

    # original frames and duration are preserved
    assert aligned_video.duration == video.duration
    for t in (0.0, 1.5, 9.9):
        assert np.array_equal(aligned_video.get_frame(t), video.get_frame(t))

    # the estimate is exposed and the audio is the shifted recording
    estimate = aligned_video.audio_alignment
    assert estimate.ok
    assert estimate.offset == pytest.approx(offset, abs=2.0 / FPS)
    tt = np.arange(2.0, 4.0, 1.0 / FPS)
    assert np.allclose(
        aligned_video.audio.to_soundarray(tt=tt, fps=FPS),
        recorded.to_soundarray(tt=tt - estimate.offset, fps=FPS),
        atol=1e-6,
    )


def test_align_audio_to_video_with_explicit_reference_and_estimate():
    offset = 0.8
    reference, recorded = make_clips(offset=offset)
    video = ColorClip((32, 24), color=(0, 0, 0), duration=REF_DURATION)

    # explicit reference (video has no audio of its own)
    aligned_video = align_audio_to_video(
        video, recorded, max_offset=2.0, reference=reference
    )
    assert aligned_video.audio_alignment.offset == pytest.approx(offset, abs=2.0 / FPS)

    # pre-computed estimate
    estimate = estimate_audio_offset(reference, recorded, max_offset=2.0)
    aligned_video = align_audio_to_video(video, recorded, estimate=estimate)
    assert aligned_video.duration == REF_DURATION


def test_align_audio_to_video_errors():
    reference, recorded = make_clips(offset=-1.3)
    video = ColorClip((32, 24), color=(0, 0, 0), duration=REF_DURATION)

    with pytest.raises(ValueError):  # no max_offset and no estimate
        align_audio_to_video(video, recorded)
    with pytest.raises(ValueError):  # no reference audio available
        align_audio_to_video(video, recorded, max_offset=2.0)

    silent = AudioArrayClip(np.zeros((int(REC_DURATION * FPS), 1)), fps=FPS)
    video = video.with_audio(reference)
    with pytest.raises(AudioAlignmentError):
        align_audio_to_video(video, silent, max_offset=2.0)
