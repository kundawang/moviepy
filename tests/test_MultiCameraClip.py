"""Tests for the multi-camera grid composition (``MultiCameraClip``)."""

import os

import numpy as np

import pytest

from moviepy import (
    AudioClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    MultiCameraClip,
    VideoClip,
    VideoFileClip,
)


# Four highly distinct colors, one per camera.
CAMERA_COLORS = [
    (255, 0, 0),  # red,    top-left
    (0, 255, 0),  # green,  top-right
    (0, 0, 255),  # blue,   bottom-left
    (255, 255, 0),  # yellow, bottom-right
]
GRID_POSITIONS = [(0, 0), (0, 1), (1, 0), (1, 1)]
# Each camera starts a little later than the previous one.
OFFSETS = [0.0, 0.25, 0.5, 0.75]
# One distinct, prime-ish pulse frequency per camera so a sample can be
# attributed back to its source.
PULSE_FREQS = [100.0, 200.0, 300.0, 400.0]

OUTPUT_SIZE = (80, 60)  # 2x2 grid of 40x30 cells
CELL_SIZE = (40, 30)
DURATION = 2.0
FPS = 10
AUDIO_FPS = 1000
BG_COLOR = (12, 34, 56)


def make_pulse(freq, duration=DURATION, fps=AUDIO_FPS):
    """A pure sine pulse whose value at time t is deterministic."""
    return AudioClip(
        lambda t, f=freq: np.sin(2 * np.pi * f * t), duration=duration, fps=fps
    )


def make_cameras(
    colors=CAMERA_COLORS,
    offsets=OFFSETS,
    size=CELL_SIZE,
    with_audio=True,
    durations=None,
    infinite=False,
):
    """Build the 4 color cameras with staggered pulse audios."""
    cameras = []
    for index, color in enumerate(colors):
        duration = None if infinite else (durations[index] if durations else DURATION)
        clip = ColorClip(size, color=color, duration=duration).with_fps(FPS)
        if with_audio and not infinite:
            clip = clip.with_audio(make_pulse(PULSE_FREQS[index], clip.duration))
        cameras.append((clip, offsets[index], GRID_POSITIONS[index]))
    return cameras


def cell_centers():
    """Center pixel (x, y) of each of the 4 grid cells."""
    cw, ch = CELL_SIZE
    return [(col * cw + cw // 2, row * ch + ch // 2) for row, col in GRID_POSITIONS]


# ---------------------------------------------------------------------------
# Construction and frame geometry
# ---------------------------------------------------------------------------


def test_grid_size_duration_and_fps():
    grid = MultiCameraClip(
        make_cameras(), size=OUTPUT_SIZE, duration=DURATION, bg_color=BG_COLOR
    )
    assert grid.size == OUTPUT_SIZE
    assert grid.duration == DURATION
    assert grid.end == DURATION
    # The highest input fps becomes the composition fps.
    assert grid.fps == FPS
    grid.close()


def test_frames_are_placed_in_the_right_cells():
    """The four solid colors must appear exactly in their declared cell."""
    grid = MultiCameraClip(
        make_cameras(), size=OUTPUT_SIZE, duration=DURATION, bg_color=BG_COLOR
    )
    # At t=1 all cameras are visible.
    frame = grid.get_frame(1.0)
    for index, (x, y) in enumerate(cell_centers()):
        np.testing.assert_array_equal(frame[y, x], CAMERA_COLORS[index])

    # The four quadrants are internally uniform.
    for (row, col), color in zip(GRID_POSITIONS, CAMERA_COLORS):
        quadrant = frame[
            row * CELL_SIZE[1] : (row + 1) * CELL_SIZE[1],
            col * CELL_SIZE[0] : (col + 1) * CELL_SIZE[0],
        ]
        assert np.all(quadrant == color)
    grid.close()


def test_missing_frame_regions_show_background_color():
    """Before a camera starts its whole cell is painted with bg_color."""
    grid = MultiCameraClip(
        make_cameras(), size=OUTPUT_SIZE, duration=DURATION, bg_color=BG_COLOR
    )
    # At t=0 only camera 0 (offset 0) is visible.
    frame = grid.get_frame(0.0)
    assert tuple(frame[cell_centers()[0][1], cell_centers()[0][0]]) == CAMERA_COLORS[0]
    for index in (1, 2, 3):
        x, y = cell_centers()[index]
        np.testing.assert_array_equal(frame[y, x], BG_COLOR)

    # At t=0.3 cameras 0 and 1 are visible, 2 and 3 still missing.
    frame = grid.get_frame(0.3)
    for index in (0, 1):
        x, y = cell_centers()[index]
        np.testing.assert_array_equal(frame[y, x], CAMERA_COLORS[index])
    for index in (2, 3):
        x, y = cell_centers()[index]
        np.testing.assert_array_equal(frame[y, x], BG_COLOR)
    grid.close()


def test_camera_that_ends_early_falls_back_to_background():
    """After a finite source ends, its cell reverts to the background."""
    durations = [DURATION, 0.3, DURATION, DURATION]
    grid = MultiCameraClip(
        make_cameras(durations=durations),
        size=OUTPUT_SIZE,
        duration=DURATION,
        bg_color=BG_COLOR,
    )
    # Camera 1 spans [0.25, 0.55); at t=1 it is long gone.
    frame = grid.get_frame(1.0)
    x, y = cell_centers()[1]
    np.testing.assert_array_equal(frame[y, x], BG_COLOR)
    # Camera 0 is still there.
    x0, y0 = cell_centers()[0]
    np.testing.assert_array_equal(frame[y0, x0], CAMERA_COLORS[0])
    grid.close()


def test_infinite_clips_are_bounded_by_output_duration():
    """Feeds with duration=None are legal as long as the output is finite."""
    grid = MultiCameraClip(
        make_cameras(infinite=True, with_audio=False),
        size=OUTPUT_SIZE,
        duration=DURATION,
        audio_mode="mix",
    )
    assert grid.duration == DURATION
    # Every frame up to the output duration can be rendered.
    for t in np.arange(0, DURATION, 1 / FPS):
        frame = grid.get_frame(t)
        assert frame.shape == (OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3)
    grid.close()


def test_dictionary_camera_specs_are_accepted():
    cameras = [
        {
            "clip": ColorClip(CELL_SIZE, color=c, duration=DURATION).with_fps(FPS),
            "offset": offset,
            "position": position,
        }
        for c, offset, position in zip(CAMERA_COLORS, OFFSETS, GRID_POSITIONS)
    ]
    grid = MultiCameraClip(
        cameras, size=OUTPUT_SIZE, duration=DURATION, audio_mode="mix"
    )
    frame = grid.get_frame(1.0)
    for index, (x, y) in enumerate(cell_centers()):
        np.testing.assert_array_equal(frame[y, x], CAMERA_COLORS[index])
    grid.close()


# ---------------------------------------------------------------------------
# fit modes: contain (letterbox) vs cover (center crop)
# ---------------------------------------------------------------------------


def test_contain_fit_letterboxes_with_background():
    # Source twice as wide as the cell -> pillarboxed vertically.
    wide = ColorClip((80, 30), color=CAMERA_COLORS[0], duration=DURATION).with_fps(FPS)
    grid = MultiCameraClip(
        [(wide, 0.0, (0, 0))],
        size=CELL_SIZE,
        duration=DURATION,
        grid_shape=(1, 1),
        fit="contain",
        bg_color=BG_COLOR,
        audio_mode="mix",
    )
    frame = grid.get_frame(0.0)
    assert frame.shape == (CELL_SIZE[1], CELL_SIZE[0], 3)
    # The scaled strip (40x15) is centered: colored middle rows.
    np.testing.assert_array_equal(
        frame[CELL_SIZE[1] // 2, CELL_SIZE[0] // 2], CAMERA_COLORS[0]
    )
    # Top and bottom borders are background.
    np.testing.assert_array_equal(frame[0, 0], BG_COLOR)
    np.testing.assert_array_equal(frame[-1, 0], BG_COLOR)
    grid.close()


def test_cover_fit_center_crops_without_background_inside_cell():
    # Source twice as wide -> scaled to 80x30 then center-cropped to 40x30.
    wide = ColorClip((80, 30), color=CAMERA_COLORS[1], duration=DURATION).with_fps(FPS)
    grid = MultiCameraClip(
        [(wide, 0.0, (0, 0))],
        size=CELL_SIZE,
        duration=DURATION,
        grid_shape=(1, 1),
        fit="cover",
        bg_color=BG_COLOR,
        audio_mode="mix",
    )
    frame = grid.get_frame(0.0)
    assert frame.shape == (CELL_SIZE[1], CELL_SIZE[0], 3)
    # The whole cell is filled with the source color, no border anywhere.
    assert np.all(frame == CAMERA_COLORS[1])
    grid.close()


def test_cover_tall_source_is_center_cropped():
    # Source twice as tall -> scaled to 40x60 then center-cropped to 40x30.
    tall = ColorClip((40, 60), color=CAMERA_COLORS[2], duration=DURATION).with_fps(FPS)
    grid = MultiCameraClip(
        [(tall, 0.0, (0, 0))],
        size=CELL_SIZE,
        duration=DURATION,
        grid_shape=(1, 1),
        fit="cover",
        audio_mode="mix",
    )
    frame = grid.get_frame(0.0)
    assert frame.shape == (CELL_SIZE[1], CELL_SIZE[0], 3)
    assert np.all(frame == CAMERA_COLORS[2])
    grid.close()


# ---------------------------------------------------------------------------
# Audio: host selection, weighted mixing, alignment, missing tracks
# ---------------------------------------------------------------------------


def test_host_audio_is_time_aligned_to_its_offset():
    host_index = 2
    grid = MultiCameraClip(
        make_cameras(),
        size=OUTPUT_SIZE,
        duration=DURATION,
        audio_mode="host",
        host_index=host_index,
    )
    offset = OFFSETS[host_index]
    freq = PULSE_FREQS[host_index]

    # Before the host starts, silence.
    early = grid.audio.get_frame(np.array([0.0, offset - 0.1]))
    assert np.allclose(early, 0.0, atol=1e-12)

    # After it starts, the sample equals the source pulse at (t - offset).
    times = np.array([offset + 0.11, offset + 0.37, 1.5])
    expected = np.sin(2 * np.pi * freq * (times - offset))
    np.testing.assert_allclose(grid.audio.get_frame(times), expected, atol=1e-12)

    # Audio spans exactly the output duration.
    assert grid.audio.duration == DURATION
    grid.close()


def test_host_audio_ignores_other_cameras():
    grid = MultiCameraClip(
        make_cameras(),
        size=OUTPUT_SIZE,
        duration=DURATION,
        audio_mode="host",
        host_index=0,
    )
    # Only the 100Hz pulse of camera 0 is present, starting at t=0.
    times = np.array([0.13, 0.41])
    np.testing.assert_allclose(
        grid.audio.get_frame(times),
        np.sin(2 * np.pi * PULSE_FREQS[0] * times),
        atol=1e-12,
    )
    grid.close()


def test_mix_audio_sums_all_aligned_weighted_pulses():
    weights = {0: 1.0, 1: 0.5, 2: 1.0, 3: 0.25}
    grid = MultiCameraClip(
        make_cameras(),
        size=OUTPUT_SIZE,
        duration=DURATION,
        audio_mode="mix",
        audio_weights=weights,
    )
    assert isinstance(grid.audio, CompositeAudioClip)
    t = 1.25  # all four cameras are playing
    expected = 0.0
    for index, (offset, freq) in enumerate(zip(OFFSETS, PULSE_FREQS)):
        expected += weights[index] * np.sin(2 * np.pi * freq * (t - offset))
    sample = grid.audio.get_frame(t)
    sample = sample[0] if np.ndim(sample) == 1 else sample
    assert np.all(sample == pytest.approx(expected, abs=1e-12))

    # Before camera 3 starts, its weighted pulse contributes nothing.
    t = 0.3  # only cameras 0 and 1 play
    expected = 1.0 * np.sin(
        2 * np.pi * PULSE_FREQS[0] * (t - OFFSETS[0])
    ) + 0.5 * np.sin(2 * np.pi * PULSE_FREQS[1] * (t - OFFSETS[1]))
    sample = grid.audio.get_frame(t)
    sample = sample[0] if np.ndim(sample) == 1 else sample
    assert np.all(sample == pytest.approx(expected, abs=1e-12))
    grid.close()


def test_mix_skips_cameras_without_audio_track():
    cameras = make_cameras(with_audio=False)
    # Only camera 1 gets a track.
    cameras[1] = (
        cameras[1][0].with_audio(make_pulse(PULSE_FREQS[1])),
        OFFSETS[1],
        GRID_POSITIONS[1],
    )
    grid = MultiCameraClip(
        cameras, size=OUTPUT_SIZE, duration=DURATION, audio_mode="mix"
    )
    times = np.array([0.5, 1.0])
    expected = np.sin(2 * np.pi * PULSE_FREQS[1] * (times - OFFSETS[1]))
    # CompositeAudioClip normalizes mono tracks to 2 channels; the second
    # channel is a zero-filled broadcast (tiny float noise around silence).
    mixed = np.asarray(grid.audio.get_frame(times))
    mono = mixed[..., 0] if mixed.ndim == 2 else mixed
    np.testing.assert_allclose(mono, expected, atol=1e-12)
    if mixed.ndim == 2:
        np.testing.assert_allclose(mixed[:, 1], 0.0, atol=1e-10)
    grid.close()


def test_mix_without_any_audio_has_no_soundtrack():
    grid = MultiCameraClip(
        make_cameras(with_audio=False),
        size=OUTPUT_SIZE,
        duration=DURATION,
        audio_mode="mix",
    )
    assert grid.audio is None
    grid.close()


def test_host_with_missing_audio_track_raises():
    grid_kwargs = dict(size=OUTPUT_SIZE, duration=DURATION)
    with pytest.raises(ValueError, match="Host camera .* has no audio"):
        MultiCameraClip(
            make_cameras(with_audio=False),
            audio_mode="host",
            host_index=0,
            **grid_kwargs,
        )


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------


def test_duplicate_grid_position_raises():
    cameras = make_cameras()
    cameras[3] = (cameras[3][0], cameras[3][1], (0, 0))  # steal camera 0 cell
    with pytest.raises(ValueError, match="Duplicate grid position"):
        MultiCameraClip(cameras, size=OUTPUT_SIZE, duration=DURATION)


def test_negative_offset_raises():
    cameras = make_cameras()
    cameras[1] = (cameras[1][0], -0.1, GRID_POSITIONS[1])
    with pytest.raises(ValueError, match="invalid offset"):
        MultiCameraClip(cameras, size=OUTPUT_SIZE, duration=DURATION)


def test_offset_beyond_duration_raises():
    cameras = make_cameras()
    cameras[1] = (cameras[1][0], DURATION, GRID_POSITIONS[1])
    with pytest.raises(ValueError, match="never be visible"):
        MultiCameraClip(cameras, size=OUTPUT_SIZE, duration=DURATION)


def test_non_divisible_size_raises():
    with pytest.raises(ValueError, match="not divisible"):
        MultiCameraClip(make_cameras(), size=(81, 60), duration=DURATION)


def test_position_outside_grid_raises():
    cameras = make_cameras()
    cameras[0] = (cameras[0][0], 0.0, (2, 0))
    with pytest.raises(ValueError, match="outside the"):
        MultiCameraClip(cameras, size=OUTPUT_SIZE, duration=DURATION)


def test_invalid_fit_and_audio_modes_raise():
    cameras = make_cameras()
    with pytest.raises(ValueError, match="fit must be"):
        MultiCameraClip(cameras, size=OUTPUT_SIZE, duration=DURATION, fit="stretch")
    with pytest.raises(ValueError, match="audio_mode must be"):
        MultiCameraClip(
            cameras, size=OUTPUT_SIZE, duration=DURATION, audio_mode="karaoke"
        )


def test_invalid_host_index_raises():
    with pytest.raises(ValueError, match="host_index"):
        MultiCameraClip(
            make_cameras(), size=OUTPUT_SIZE, duration=DURATION, host_index=4
        )


def test_invalid_audio_weight_raises():
    with pytest.raises(ValueError, match="non-negative"):
        MultiCameraClip(
            make_cameras(),
            size=OUTPUT_SIZE,
            duration=DURATION,
            audio_mode="mix",
            audio_weights={0: -1.0},
        )


def test_clip_without_size_raises():
    cameras = make_cameras()
    cameras[0] = (VideoClip(), 0.0, (0, 0))
    with pytest.raises(ValueError, match="no size attribute"):
        MultiCameraClip(cameras, size=OUTPUT_SIZE, duration=DURATION)


def test_non_positive_duration_raises():
    with pytest.raises(ValueError, match="duration must be"):
        MultiCameraClip(make_cameras(), size=OUTPUT_SIZE, duration=0)


# ---------------------------------------------------------------------------
# Equivalence with a hand-built CompositeVideoClip ("original composition")
# ---------------------------------------------------------------------------


def test_equivalent_to_manual_composite_video_and_audio():
    """MultiCameraClip must match a composition built from the same primitives."""
    cameras = make_cameras()
    grid = MultiCameraClip(
        cameras,
        size=OUTPUT_SIZE,
        duration=DURATION,
        fit="contain",
        bg_color=(0, 0, 0),
        audio_mode="mix",
    )

    # Manual, reference composition using only the base compositing tools.
    manual_cells = []
    manual_tracks = []
    for index, (clip, offset, (row, col)) in enumerate(cameras):
        cell = CompositeVideoClip(
            [clip.without_audio().with_position("center")],
            size=CELL_SIZE,
            bg_color=(0, 0, 0),
        )
        cell = (
            cell.with_duration(DURATION)
            .with_start(offset)
            .with_position((col * CELL_SIZE[0], row * CELL_SIZE[1]))
        )
        manual_cells.append(cell)
        manual_tracks.append(clip.audio.subclipped(0, DURATION).with_start(offset))
    manual = CompositeVideoClip(
        manual_cells, size=OUTPUT_SIZE, bg_color=(0, 0, 0)
    ).with_duration(DURATION)
    manual.audio = CompositeAudioClip(manual_tracks).with_duration(DURATION)

    for t in np.arange(0, DURATION, 1 / FPS):
        np.testing.assert_array_equal(grid.get_frame(t), manual.get_frame(t))
    audio_times = np.linspace(0, DURATION - 1 / AUDIO_FPS, 37)
    np.testing.assert_allclose(
        grid.audio.get_frame(audio_times),
        manual.audio.get_frame(audio_times),
        atol=1e-12,
    )
    manual.close()
    grid.close()


# ---------------------------------------------------------------------------
# Resource management and end-to-end rendering
# ---------------------------------------------------------------------------


def test_close_does_not_close_owned_source_clips(util):
    """Closing the grid must release internals but leave source clips usable."""
    path = os.path.join(util.TMP_DIR, "multicam_source.mp4")
    source = ColorClip(CELL_SIZE, color=CAMERA_COLORS[0], duration=1).with_fps(FPS)
    source.write_videofile(path, logger=None, preset="ultrafast")

    file_clip = VideoFileClip(path)
    grid = MultiCameraClip(
        [(file_clip, 0.0, (0, 0))],
        size=CELL_SIZE,
        duration=1.0,
        grid_shape=(1, 1),
        audio_mode="mix",
    )
    grid.get_frame(0.5)
    grid.close()

    # The reader of the caller-owned source clip must still be alive.
    assert file_clip.reader is not None
    file_clip.get_frame(0.1)
    file_clip.close()


def test_context_manager_closes_without_leak():
    grid = MultiCameraClip(
        make_cameras(), size=OUTPUT_SIZE, duration=DURATION, bg_color=BG_COLOR
    )
    with grid as entered:
        assert entered is grid
        entered.get_frame(0.5)
    # Internally created background and audio are released.
    assert grid.bg is None
    assert grid.audio is None


def test_end_to_end_write_and_readback(util):
    output = os.path.join(util.TMP_DIR, "multicam_grid.mp4")
    grid = MultiCameraClip(
        make_cameras(), size=OUTPUT_SIZE, duration=DURATION, bg_color=BG_COLOR
    )
    grid.write_videofile(output, fps=FPS, logger=None, preset="ultrafast")
    grid.close()

    readback = VideoFileClip(output)
    assert readback.size == list(OUTPUT_SIZE)
    assert readback.duration == pytest.approx(DURATION, abs=0.05)
    # At t=1 every quadrant carries its camera color (codec tolerance).
    frame = readback.get_frame(1.0)
    for index, (x, y) in enumerate(cell_centers()):
        np.testing.assert_allclose(frame[y, x], CAMERA_COLORS[index], atol=8)
    assert readback.audio is not None
    readback.close()


if __name__ == "__main__":
    pytest.main()
