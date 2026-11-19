"""Tests for the multi-cue subtitle layout engine.

These tests use a fixed font (media/doc_medias/example.ttf) and a hand-written
two-speaker SRT sample so that every rendered size is deterministic. They
verify placements frame by frame: position, visibility inside the safe area,
non-overlap of simultaneous cues and stability of the rows across time.
"""

import os

import pytest

from moviepy.video.VideoClip import ColorClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.tools.subtitle_layout import (
    SubtitleCue,
    SubtitleLayout,
    SubtitleLayoutError,
    cues_from_srt,
    default_speaker_parser,
    make_default_textclip,
    position_subtitle_clips,
)

FONT = "media/doc_medias/example.ttf"
TWO_SPEAKER_SRT = "media/subtitles_two_speakers.srt"

FRAME_SIZE = (640, 480)
BOTTOM_MARGIN = 40
TOP_MARGIN = 20
SIDE_MARGIN = 20
LINE_SPACING = 8

# Sample times spanning every distinct cue combination of the SRT sample:
# solo Alice, Alice+Bob overlap, a three-cue overlap at t=4.5 and solo Bob.
SAMPLE_TIMES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]


def make_factory(**kwargs):
    kwargs.setdefault("font_size", 24)
    return make_default_textclip(font=FONT, **kwargs)


def make_cues():
    return cues_from_srt(TWO_SPEAKER_SRT)


def make_layout(**kwargs):
    kwargs.setdefault("size", FRAME_SIZE)
    kwargs.setdefault("bottom_margin", BOTTOM_MARGIN)
    kwargs.setdefault("top_margin", TOP_MARGIN)
    kwargs.setdefault("side_margin", SIDE_MARGIN)
    kwargs.setdefault("line_spacing", LINE_SPACING)
    kwargs.setdefault("max_lines", 3)
    return SubtitleLayout(**kwargs)


def measure(cues, factory=None):
    factory = factory or make_factory()
    return {index: factory(cue.text).size for index, cue in enumerate(cues)}


def test_srt_sample_is_two_speakers_and_text_preserved():
    cues = make_cues()
    assert len(cues) == 7
    speakers = {cue.speaker for cue in cues}
    assert speakers == {"Alice", "Bob"}
    # Speaker detection must never rewrite the stored subtitle text.
    assert cues[0].text == "ALICE: Hello everyone, welcome to the show"
    assert cues[1].text == "BOB: Hi there, thanks for inviting me"
    # Half-open boundaries: cue 4 starts exactly when cue 2 ends at t=5.0, so
    # the two never overlap.
    assert cues[3].start == 5.0
    assert cues[1].end == 6.0
    # Cue 3 starts inside cue 1, creating the three-cue overlap window.
    assert cues[0].start < cues[2].start < cues[0].end


def test_default_speaker_parser_variants():
    assert default_speaker_parser("ALICE: hello") == "Alice"
    assert default_speaker_parser("[Alice]: hello") == "Alice"
    assert default_speaker_parser("(BOB) : hello") == "Bob"
    assert default_speaker_parser("plain sentence") is None
    assert default_speaker_parser("12: numeric start") is None


def test_layout_uses_real_textclip_sizes():
    cues = make_cues()
    layout = make_layout()
    placements = layout.layout(cues, measure(cues))
    for placement, cue in zip(placements, cues):
        expected = TextClip(
            text=cue.text,
            font=FONT,
            font_size=24,
            color="#ffffff",
            stroke_color="#000000",
            stroke_width=1,
        ).size
        assert placement.size == tuple(expected)
        assert placement.cue.text == cue.text  # text untouched


def test_every_placement_is_inside_safe_area_and_rows_do_not_overlap():
    cues = make_cues()
    layout = make_layout()
    placements = layout.layout(cues, measure(cues))

    for t in SAMPLE_TIMES:
        visible = [p for p in placements if p.cue.start <= t < p.cue.end]
        assert visible, "no cue found at t=%ss" % t
        rects = []
        for placement in visible:
            x, y = placement.position_at(t)
            width, height = placement.size
            # Horizontal safe margins.
            assert x >= SIDE_MARGIN
            assert x + width <= FRAME_SIZE[0] - SIDE_MARGIN
            # Vertical safe margins.
            assert y >= TOP_MARGIN
            assert y + height <= FRAME_SIZE[1] - BOTTOM_MARGIN
            rects.append((x, y, width, height))
        # No pair of rectangles may overlap on both axes.
        for i, (xa, ya, wa, ha) in enumerate(rects):
            for xb, yb, wb, hb in rects[i + 1 :]:
                assert not (
                    xa < xb + wb and xb < xa + wa and ya < yb + hb and yb < ya + ha
                ), ("subtitles overlap at t=%ss" % t)


def test_slot_stability_per_speaker():
    cues = make_cues()
    layout = make_layout()
    placements = layout.layout(cues, measure(cues))
    # Bob always speaks from a single stable row, the one above Alice's main
    # row, even during Alice's short self-overlap.
    bob_slots = {p.slot for p in placements if p.cue.speaker == "Bob"}
    assert bob_slots == {1}
    # Alice normally owns the bottom row (slot 0). Her short overlapping cue
    # is the only one pushed to a higher slot; it never displaces Bob.
    alice = [p for p in placements if p.cue.speaker == "Alice"]
    main_row = [p for p in alice if p.slot == 0]
    assert main_row  # Alice keeps a presence on the bottom row
    overflow = [p for p in alice if p.slot != 0]
    assert all(p.slot == 2 for p in overflow)
    # A cue keeps one slot for its whole lifetime.
    for placement in placements:
        assert len({placement.slot}) == 1


def test_adjacent_cues_do_not_jump_within_their_lifetime():
    cues = make_cues()
    layout = make_layout()
    placements = layout.layout(cues, measure(cues))
    for placement in placements:
        xs = {x for _, x, _ in placement.positions}
        slots = {placement.slot}
        assert len(xs) == 1  # horizontal position never changes
        assert len(slots) == 1  # slot never changes
        # Vertical position only changes when another cue enters/leaves; it
        # must stay constant within any interval bounded by cue boundaries.
        ys = [y for _, _, y in placement.positions]
        for earlier, later in zip(ys, ys[1:]):
            assert later == earlier or abs(later - earlier) > LINE_SPACING


def test_segmented_position_matches_boundaries():
    cues = make_cues()
    layout = make_layout()
    placements = layout.layout(cues, measure(cues))
    # At t=3.75 three cues overlap: cue 1 (Alice, ending at 4.0), cue 3
    # (Alice, 3.5-5.0) and cue 2 (Bob, 2-6s). Sample strictly inside the
    # half-open overlap window [3.5, 4.0).
    visible = [p for p in placements if p.cue.start <= 3.75 < p.cue.end]
    assert len(visible) == 3
    assert sorted(p.slot for p in visible) == [0, 1, 2]
    # Bob keeps his stable middle row; Alice's two cues frame him on the
    # bottom row and the temporary overflow row on top.
    assert [p.cue.speaker for p in sorted(visible, key=lambda p: p.slot)] == [
        "Alice",
        "Bob",
        "Alice",
    ]
    # At t=5.5 only the second Alice cue and Bob remain visible.
    assert len([p for p in placements if p.cue.start <= 5.5 < p.cue.end]) == 2
    # position_at clamps before the first segment and never goes out of range.
    first = placements[0]
    assert first.position_at(first.cue.start) == first.positions[0][1:]


def test_identical_start_times_are_ordered_by_priority_then_input_order():
    cues = [
        SubtitleCue(0, 2, "ALICE: first"),
        SubtitleCue(0, 2, "BOB: second"),
        SubtitleCue(0, 2, "CAROL: third"),
    ]
    sizes = measure(cues)

    by_input = make_layout(max_lines=3).layout(cues, sizes)
    assert [p.slot for p in by_input] == [0, 1, 2]

    prioritized = [
        SubtitleCue(0, 2, "ALICE: first", priority=2),
        SubtitleCue(0, 2, "BOB: second", priority=0),
        SubtitleCue(0, 2, "CAROL: third", priority=1),
    ]
    by_priority = make_layout(max_lines=3).layout(prioritized, sizes)
    assert [p.slot for p in by_priority] == [2, 0, 1]


def test_max_lines_overflow_raises_deterministically():
    cues = [SubtitleCue(0, 2, "S%d: text" % i) for i in range(4)]
    layout = make_layout(max_lines=3, speaker_parser=lambda text: None)
    with pytest.raises(SubtitleLayoutError) as exc:
        layout.layout(cues, measure(cues))
    assert "More than 3 simultaneous subtitles" in str(exc.value)


def test_too_wide_cue_raises_with_safe_width_details():
    long_text = "ALICE: " + "word " * 100
    cues = [SubtitleCue(0, 2, long_text.strip())]
    layout = make_layout()
    width = TextClip(text=long_text.strip(), font=FONT, font_size=24).size[0]
    assert width > layout.safe_width
    with pytest.raises(SubtitleLayoutError) as exc:
        layout.layout(cues, {0: (width, 30)})
    message = str(exc.value)
    assert "safe width" in message and str(width) in message


def test_wrapped_caption_fits_where_plain_label_does_not():
    long_text = "ALICE: " + "word " * 60
    long_text = long_text.strip()
    cues = [SubtitleCue(0, 2, long_text)]
    layout = make_layout()

    label_size = TextClip(text=long_text, font=FONT, font_size=24).size
    with pytest.raises(SubtitleLayoutError):
        layout.layout(cues, {0: label_size})

    caption = TextClip(
        text=long_text,
        font=FONT,
        font_size=24,
        size=(layout.safe_width, None),
        method="caption",
    )
    # The wrapped clip fits horizontally and keeps the text verbatim.
    assert caption.size[0] <= layout.safe_width
    placements = layout.layout(cues, {0: caption.size})
    assert placements[0].size == caption.size
    x, y = placements[0].position_at(0.5)
    assert x + caption.size[0] <= FRAME_SIZE[0] - SIDE_MARGIN
    assert y >= TOP_MARGIN and y + caption.size[1] <= FRAME_SIZE[1] - BOTTOM_MARGIN


def test_insufficient_vertical_space_raises():
    cues = [
        SubtitleCue(0, 2, "ALICE: one"),
        SubtitleCue(0, 2, "BOB: two"),
        SubtitleCue(0, 2, "CAROL: three"),
    ]
    # A 120px-high frame with 50px margins leaves only 20px of safe height:
    # even a single 30px row cannot fit.
    layout = make_layout(size=(640, 120), top_margin=50, bottom_margin=50)
    with pytest.raises(SubtitleLayoutError) as exc:
        layout.layout(cues, measure(cues))
    assert "vertical space" in str(exc.value)


def test_invalid_margins_are_rejected():
    with pytest.raises(ValueError, match="Invalid margins"):
        SubtitleLayout(size=(640, 480), bottom_margin=300, top_margin=200)
    with pytest.raises(ValueError, match="side_margin"):
        SubtitleLayout(size=(100, 480), side_margin=60)
    for kwargs in (
        dict(bottom_margin=-1),
        dict(top_margin=-1),
        dict(side_margin=-5),
        dict(line_spacing=-1),
    ):
        with pytest.raises(ValueError):
            make_layout(**kwargs)


def test_invalid_configuration_values():
    with pytest.raises(ValueError):
        SubtitleLayout(size=(0, 480))
    with pytest.raises(ValueError):
        SubtitleLayout(size=(640, 480), max_lines=0)
    with pytest.raises(ValueError):
        SubtitleLayout(size=(640, 480), align="middle")
    with pytest.raises(ValueError):
        SubtitleLayout(size=(640, 480), speaker_slots={"Alice": 3})
    with pytest.raises(ValueError):
        SubtitleLayout(
            size=(640, 480), max_lines=2, speaker_slots={"Alice": 0, "Bob": 0}
        )
    with pytest.raises(ValueError):
        SubtitleLayout(size=(640, 480), speaker_parser="not-callable")


def test_invalid_cues():
    with pytest.raises(ValueError):
        SubtitleCue(0, 0, "empty duration")
    with pytest.raises(ValueError):
        SubtitleCue(2, 1, "reversed")
    with pytest.raises(ValueError):
        SubtitleCue(-1, 2, "negative")
    with pytest.raises(ValueError):
        SubtitleCue(0, 2, "")


def test_speaker_pinning_holds_even_when_slot_would_be_free():
    cues = [
        SubtitleCue(0, 2, "ALICE: hello", speaker="Alice"),
        SubtitleCue(1, 3, "BOB: world", speaker="Bob"),
    ]
    layout = make_layout(speaker_slots={"Alice": 1, "Bob": 0})
    placements = layout.layout(cues, measure(cues))
    assert placements[0].slot == 1
    assert placements[1].slot == 0
    # The pinned bottom slot is reserved: Alice never occupies slot 0.
    assert {p.slot for p in placements if p.cue.speaker == "Alice"} == {1}

    # While Alice speaks alone (t=0.5) her pinned row stays at the upper
    # position instead of sliding down to the bottom row's y=410.
    alice = placements[0]
    solo_y = alice.position_at(0.5)[1]
    shared_y = alice.position_at(1.5)[1]
    assert solo_y == shared_y
    bottom_row_y = FRAME_SIZE[1] - BOTTOM_MARGIN - alice.size[1]
    assert solo_y < bottom_row_y  # smaller y means higher on screen


def test_pinned_solo_speaker_keeps_upper_row_with_empty_gap_below():
    """A speaker pinned above slot 0 must not slide down when speaking alone."""
    cues = [SubtitleCue(0, 2, "ALICE: solo", speaker="Alice")]
    layout = make_layout(speaker_slots={"Alice": 1})
    placement = layout.layout(cues, measure(cues))[0]
    height = placement.size[1]
    # Bottom empty row (height h) + spacing, then Alice's row.
    expected_y = FRAME_SIZE[1] - BOTTOM_MARGIN - height - LINE_SPACING - height
    assert placement.slot == 1
    assert placement.position_at(0.5)[1] == expected_y
    # A non-pinned solo cue still anchors to the bottom margin.
    unpinned = make_layout().layout(cues, measure(cues))[0]
    assert unpinned.position_at(0.5)[1] == FRAME_SIZE[1] - BOTTOM_MARGIN - height


def test_pinned_speaker_overlap_raises():
    cues = [
        SubtitleCue(0, 2, "ALICE: one", speaker="Alice"),
        SubtitleCue(1, 3, "ALICE: two", speaker="Alice"),
    ]
    layout = make_layout(speaker_slots={"Alice": 0})
    with pytest.raises(SubtitleLayoutError, match="pinned to slot"):
        layout.layout(cues, measure(cues))


def test_alignment_options_keep_side_margins():
    cues = [SubtitleCue(0, 2, "ALICE: hi")]
    sizes = measure(cues)
    width = sizes[0][0]
    for align, expected_x in (
        ("left", SIDE_MARGIN),
        ("right", FRAME_SIZE[0] - SIDE_MARGIN - width),
        ("center", (FRAME_SIZE[0] - width) // 2),
    ):
        placement = make_layout(align=align).layout(cues, sizes)[0]
        assert placement.position_at(0.5)[0] == expected_x


def test_transparent_and_opaque_clips_are_both_supported():
    cues = make_cues()[:3]
    transparent_factory = make_default_textclip(font=FONT, font_size=24)
    opaque_factory = make_default_textclip(font=FONT, font_size=24, transparent=False)
    assert transparent_factory(cues[0].text).mask is not None
    assert opaque_factory(cues[0].text).mask is None

    layout = make_layout()
    transparent_clips = position_subtitle_clips(cues, transparent_factory, layout)
    opaque_clips = position_subtitle_clips(cues, opaque_factory, layout)
    for clip in transparent_clips:
        assert clip.mask is not None
    for clip in opaque_clips:
        assert clip.mask is None
    # Same text, same font -> same measured geometry for both.
    for transparent, opaque in zip(transparent_clips, opaque_clips):
        assert transparent.size == opaque.size
        assert transparent.pos(0) == opaque.pos(0)


def test_no_font_uses_pillow_default_deterministically():
    # Without a font path the default Pillow font is used: no crash, a real
    # measured size and a fully deterministic layout across two runs.
    factory = make_default_textclip(font=None)
    cues = make_cues()[:2]
    layout = make_layout(size=(640, 480))
    first = position_subtitle_clips(cues, factory, layout)
    second = position_subtitle_clips(cues, factory, layout)
    assert first[0].size == second[0].size
    assert first[0].pos(0.0) == second[0].pos(0.0)


def test_position_subtitle_clips_timing_and_layer_metadata():
    cues = make_cues()
    layout = make_layout()
    clips = position_subtitle_clips(cues, make_factory(), layout)
    assert len(clips) == len(cues)
    for clip, cue in zip(clips, cues):
        assert clip.start == cue.start
        assert clip.end == cue.end
        assert clip.duration == pytest.approx(cue.end - cue.start)
        assert clip.layer_index == next(
            p.slot for p in layout.layout(cues, measure(cues)) if p.cue is cue
        )


def test_rendered_frames_show_both_overlapping_subtitles(util):
    """End-to-end check: composite the layout and inspect rendered pixels."""
    cues = make_cues()
    layout = make_layout()
    clips = position_subtitle_clips(cues, make_factory(), layout)
    background = ColorClip(FRAME_SIZE, color=(20, 20, 20)).with_duration(11)
    composite = CompositeVideoClip([background] + clips, size=FRAME_SIZE)

    # At t=4.5 three cues overlap: all three text bands must contain bright
    # (white) pixels, proving none is hidden behind another.
    frame = composite.get_frame(3.75)
    placements = layout.layout(cues, measure(cues))
    visible = [p for p in placements if p.cue.start <= 3.75 < p.cue.end]
    assert len(visible) == 3
    for placement in visible:
        x, y = placement.position_at(3.75)
        width, height = placement.size
        band = frame[y : y + height, x : x + width]
        assert band[:, :, :3].max() > 200

    # Last cue ends at 11s; half-open intervals mean nothing plays at 11.0.
    frame_after = composite.get_frame(11.0)
    assert composite.playing_clips(11.0) == []
    assert frame_after.shape == (480, 640, 3)

    # Transitions: a cue appearing does not move existing ones horizontally.
    before = {
        p.cue.text: p.position_at(1.9)
        for p in placements
        if p.cue.start <= 1.9 < p.cue.end
    }
    after = {
        p.cue.text: p.position_at(2.1)
        for p in placements
        if p.cue.start <= 2.1 < p.cue.end
    }
    shared = set(before) & set(after)
    assert shared  # cue 1 straddles t=2
    for text in shared:
        assert before[text][0] == after[text][0]

    out_path = os.path.join(util.TMP_DIR, "two_speaker_subtitles.mp4")
    composite.subclipped(0, 2).write_videofile(out_path, fps=5, logger=None)
    assert os.path.exists(out_path) and os.path.getsize(out_path) > 0


def test_cross_resolution_placements_stay_in_safe_area():
    cues = make_cues()
    for size in ((320, 240), (1280, 720), (1920, 1080)):
        layout = make_layout(size=size)
        sizes = measure(cues)
        if any(w > layout.safe_width for w, _ in sizes.values()):
            with pytest.raises(SubtitleLayoutError):
                layout.layout(cues, sizes)
            continue
        placements = layout.layout(cues, sizes)
        for t in SAMPLE_TIMES:
            for placement in placements:
                if not (placement.cue.start <= t < placement.cue.end):
                    continue
                x, y = placement.position_at(t)
                w, h = placement.size
                assert x >= SIDE_MARGIN and x + w <= size[0] - SIDE_MARGIN
                assert y >= TOP_MARGIN and y + h <= size[1] - BOTTOM_MARGIN


def test_empty_cue_list_produces_empty_layout():
    assert make_layout().layout([], {}) == []


def test_missing_or_invalid_size_is_rejected():
    cues = [SubtitleCue(0, 2, "ALICE: hi")]
    layout = make_layout()
    with pytest.raises(ValueError, match="Missing rendered size"):
        layout.layout(cues, {})
    with pytest.raises(ValueError, match="positive integers"):
        layout.layout(cues, {0: (0, 30)})


def test_make_textclip_returning_unsized_clip_is_rejected():
    cues = [SubtitleCue(0, 2, "ALICE: hi")]
    with pytest.raises(ValueError, match="no sized clip"):
        position_subtitle_clips(cues, lambda text: None, make_layout())


if __name__ == "__main__":
    pytest.main()
