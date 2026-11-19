"""Multi-cue subtitle layout.

The classic :class:`~moviepy.video.tools.subtitles.SubtitlesClip` draws every
cue at the same position: when two people speak at the same time their texts
overlap and become unreadable. This module solves the *layout* part of the
problem: it assigns every simultaneous cue its own non-overlapping row
(called a *slot*) inside a bottom safe area, based on the real pixel size of
each rendered :class:`~moviepy.video.VideoClip.TextClip`.

The layout is computed once, ahead of rendering, from the full cue list:

* cues are ordered deterministically and assigned slots with a stable
  algorithm, so a speaker keeps the same row across adjacent cues instead of
  jumping around;
* every placement is checked against the frame size, the safe margins and the
  configured maximum number of rows. Overflow never silently clips text:
  :class:`SubtitleLayoutError` is raised instead;
* subtitle text is never modified, only read in order to measure it.

The main entry points are :class:`SubtitleLayout`, which produces plain
:class:`SubtitlePlacement` records, and :func:`position_subtitle_clips`, which
turns the same cues into ready-to-composite timed ``TextClip`` instances.
"""

import bisect
import re
from dataclasses import dataclass

from moviepy.video.VideoClip import TextClip
from moviepy.video.tools.subtitles import file_to_subtitles

_USE_DEFAULT_PARSER = object()


class SubtitleLayoutError(ValueError):
    """Raised when cues cannot be placed inside the configured safe area.

    This is a :class:`ValueError` subclass, so callers that already catch
    ``ValueError`` keep working. It is raised deterministically at layout
    computation time (never mid-render) and its message always states which
    cue failed, at what time and why.
    """


# A leading ``ALICE:`` / ``Alice :`` marker, optionally wrapped in brackets
# such as ``[Alice]`` or ``(ALICE)``. The marker is only used to identify a
# speaker for slot stability; the cue text is never rewritten.
_SPEAKER_PATTERN = re.compile(
    r"^\s*(?:[\[(]\s*)?([A-Za-z][\w .'-]{0,40}?)\s*[\])]?\s*:"
)


def default_speaker_parser(text):
    """Extract a speaker name from a cue such as ``"ALICE: Hello"``.

    Returns the capitalized name (``"Alice"``) or ``None`` when the text does
    not start with a ``Name:`` style marker. Bracketed markers such as
    ``"[Alice]: Hi"`` are accepted too.
    """
    match = _SPEAKER_PATTERN.match(text)
    if match is None:
        return None
    return match.group(1).strip().capitalize()


def make_default_textclip(font=None, **textclip_kwargs):
    """Build a ``text -> TextClip`` factory for use with the layout helpers.

    Parameters
    ----------
    font : str or None
        Path to a font file. When ``None`` Pillow's built-in default font is
        used, so the factory still works deterministically without any font
        installed (at a fixed small size).
    **textclip_kwargs
        Extra :class:`~moviepy.video.VideoClip.TextClip` keyword arguments
        (``font_size``, ``color``, ``method``, ``size`` ...). Note that
        ``method="caption"`` requires an explicit width and wraps long text.
    """
    if font is not None and not isinstance(font, str):
        raise ValueError("font must be a path string or None.")

    textclip_kwargs.setdefault("font_size", 24)
    textclip_kwargs.setdefault("color", "#ffffff")
    textclip_kwargs.setdefault("stroke_color", "#000000")
    textclip_kwargs.setdefault("stroke_width", 1)

    def make_textclip(text):
        return TextClip(text=text, font=font, **textclip_kwargs)

    return make_textclip


@dataclass(frozen=True)
class SubtitleCue:
    """One subtitle cue: a time interval and the text shown during it.

    Attributes
    ----------
    start : float
        Start time in seconds.
    end : float
        End time in seconds (exclusive).
    text : str
        Subtitle text. It is stored and rendered verbatim; the layout never
        modifies it.
    speaker : str or None
        Optional speaker identifier used for slot stability. When ``None`` it
        is derived from the text through ``speaker_parser``.
    priority : int or None
        Optional ranking used to break ties when several cues start at the
        same time: lower values are placed first (closer to the bottom). When
        ``None`` the order found in the source is used.
    """

    start: float
    end: float
    text: str
    speaker: str = None
    priority: int = None

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise ValueError("SubtitleCue text must be a string.")
        if self.text == "":
            raise ValueError("SubtitleCue text cannot be empty.")
        if not _is_number(self.start) or not _is_number(self.end):
            raise ValueError("SubtitleCue start and end must be numbers.")
        if self.start < 0 or self.end < 0:
            raise ValueError("SubtitleCue times cannot be negative.")
        if self.end <= self.start:
            raise ValueError(
                "SubtitleCue %r has no positive duration (start=%s, end=%s)."
                % (self.text, self.start, self.end)
            )


@dataclass(frozen=True)
class SubtitlePlacement:
    """Where and when one cue is displayed.

    A cue keeps the same slot for its whole lifetime, but its absolute
    position can change at another cue's boundary when the set of visible
    rows changes, so ``positions`` is a time-segmented list.

    Attributes
    ----------
    cue : SubtitleCue
        The cue being placed.
    slot : int
        Slot index, ``0`` being the bottom row.
    positions : tuple of (float, float, float)
        Sorted ``(time, x, y)`` segments in *absolute* timeline time. The
        placement is valid from ``time`` (inclusive) until the next segment.
        Coordinates locate the top-left corner of the text clip in the frame.
    size : tuple of (int, int)
        Measured ``(width, height)`` in pixels of the rendered text clip.
    """

    cue: SubtitleCue
    slot: int
    positions: tuple
    size: tuple

    def position_at(self, t):
        """Return the ``(x, y)`` position of the cue at absolute time ``t``."""
        times = [segment[0] for segment in self.positions]
        index = bisect.bisect_right(times, t) - 1
        index = max(index, 0)
        _, x, y = self.positions[index]
        return x, y


@dataclass(frozen=True)
class SubtitleLayout:
    """Configuration and computation of a non-overlapping subtitle layout.

    Parameters
    ----------
    size : tuple
        Frame size ``(width, height)`` in pixels.
    bottom_margin : int
        Safe distance in pixels between the bottom of the lowest subtitle and
        the bottom of the frame.
    top_margin : int
        Safe distance in pixels between the top of the highest subtitle and
        the top of the frame.
    side_margin : int
        Minimum horizontal distance kept on both sides of every subtitle.
    line_spacing : int
        Vertical gap in pixels between two consecutive rows.
    max_lines : int
        Maximum number of simultaneous subtitle rows (slots). Must be >= 1.
    align : str
        Horizontal alignment of every row: ``"center"`` (default), ``"left"``
        or ``"right"``.
    speaker_slots : dict or None
        Optional mapping ``speaker -> slot index`` pinning a speaker to a
        fixed row for the whole video. Slot ``0`` is the bottom row. Pinned
        slots are reserved: other speakers are never placed there.
    speaker_parser : callable, None or sentinel
        Function ``text -> speaker name`` used when a cue has no explicit
        ``speaker``. Defaults to :func:`default_speaker_parser`; pass ``None``
        to disable detection entirely.

    Methods
    -------
    layout(cues, text_sizes)
        Compute :class:`SubtitlePlacement` records from measured clip sizes.
    """

    size: tuple
    bottom_margin: int = 40
    top_margin: int = 20
    side_margin: int = 20
    line_spacing: int = 8
    max_lines: int = 3
    align: str = "center"
    speaker_slots: dict = None
    speaker_parser: object = _USE_DEFAULT_PARSER

    def __post_init__(self):
        width, height = self.size
        if not _is_int(width) or not _is_int(height) or width <= 0 or height <= 0:
            raise ValueError(
                "Frame size must be a tuple of two positive integers, got %r."
                % (self.size,)
            )
        for name, value in (
            ("bottom_margin", self.bottom_margin),
            ("top_margin", self.top_margin),
            ("side_margin", self.side_margin),
            ("line_spacing", self.line_spacing),
        ):
            if not _is_int(value) or value < 0:
                raise ValueError(
                    "%s must be a non-negative integer, got %r." % (name, value)
                )
        if not _is_int(self.max_lines) or self.max_lines < 1:
            raise ValueError(
                "max_lines must be a positive integer, got %r." % (self.max_lines,)
            )
        if self.align not in ("center", "left", "right"):
            raise ValueError(
                "align must be one of 'center', 'left', 'right', got %r."
                % (self.align,)
            )
        if self.bottom_margin + self.top_margin >= height:
            raise ValueError(
                "Invalid margins: bottom_margin (%d) + top_margin (%d) leave no "
                "vertical space in a frame of height %d."
                % (self.bottom_margin, self.top_margin, height)
            )
        if 2 * self.side_margin >= width:
            raise ValueError(
                "Invalid side_margin %d: the two side margins leave no horizontal "
                "space in a frame of width %d." % (self.side_margin, width)
            )
        if self.speaker_slots is not None:
            self._validate_speaker_slots(self.speaker_slots)
        if (
            self.speaker_parser is not _USE_DEFAULT_PARSER
            and self.speaker_parser is not None
            and not callable(self.speaker_parser)
        ):
            raise ValueError("speaker_parser must be callable, None or omitted.")

    def _validate_speaker_slots(self, speaker_slots):
        if not isinstance(speaker_slots, dict) or not speaker_slots:
            raise ValueError("speaker_slots must be a non-empty dict or None.")
        seen = set()
        for speaker, slot in speaker_slots.items():
            if not isinstance(speaker, str) or not speaker:
                raise ValueError("speaker_slots keys must be non-empty strings.")
            if not _is_int(slot) or slot < 0 or slot >= self.max_lines:
                raise ValueError(
                    "speaker_slots[%r] = %r is out of range: slots must be "
                    "integers in [0, max_lines=%d[." % (speaker, slot, self.max_lines)
                )
            if slot in seen:
                raise ValueError("speaker_slots assigns slot %d more than once." % slot)
            seen.add(slot)

    @property
    def safe_width(self):
        """Horizontal space available between the side margins."""
        return self.size[0] - 2 * self.side_margin

    @property
    def safe_height(self):
        """Vertical space available between the top and bottom margins."""
        return self.size[1] - self.bottom_margin - self.top_margin

    def layout(self, cues, text_sizes):
        """Compute placements for ``cues`` given their rendered clip sizes.

        Parameters
        ----------
        cues : iterable of SubtitleCue
            Cues to lay out.
        text_sizes : dict
            Mapping ``cue index (in the iterable order) -> (width, height)``
            of the corresponding rendered text clip, in pixels.

        Returns
        -------
        list of SubtitlePlacement
            One placement per cue, in the input order.

        Raises
        ------
        SubtitleLayoutError
            If more than ``max_lines`` cues overlap, a cue is too wide for the
            safe width, or the rows do not fit vertically inside the safe
            area.
        """
        cues = list(cues)
        if not cues:
            return []
        sizes = [
            self._checked_size(cues, index, text_sizes) for index in range(len(cues))
        ]

        ordered = self._ordered_cues(cues)
        slots = self._assign_slots(cues, ordered)
        return self._compute_positions(cues, sizes, slots)

    def _checked_size(self, cues, index, text_sizes):
        if index not in text_sizes:
            raise ValueError(
                "Missing rendered size for cue %d (%r)." % (index, cues[index].text)
            )
        width, height = text_sizes[index]
        if not _is_int(width) or not _is_int(height) or width <= 0 or height <= 0:
            raise ValueError(
                "Text clip size for cue %d (%r) must be two positive integers, "
                "got %r." % (index, cues[index].text, (width, height))
            )
        if width > self.safe_width:
            raise SubtitleLayoutError(
                "Cue %r at t=%ss is %dpx wide but the safe width is only %dpx "
                "(frame width %dpx, side margins %dpx). Shorten the text, reduce "
                "the font size or wrap it with a caption TextClip."
                % (
                    cues[index].text,
                    cues[index].start,
                    width,
                    self.safe_width,
                    self.size[0],
                    self.side_margin,
                )
            )
        return width, height

    def _ordered_cues(self, cues):
        """Deterministic processing order: start, then priority, then index."""

        def sort_key(item):
            index, cue = item
            priority = cue.priority if cue.priority is not None else index
            return (cue.start, priority, index)

        return [index for index, _ in sorted(enumerate(cues), key=sort_key)]

    def _resolve_speaker(self, cue):
        if cue.speaker is not None:
            return cue.speaker
        parser = self.speaker_parser
        if parser is _USE_DEFAULT_PARSER:
            parser = default_speaker_parser
        if parser is None:
            return None
        return parser(cue.text)

    def _assign_slots(self, cues, ordered):
        """Greedy, stability-first slot assignment.

        A cue keeps a slot for its whole lifetime. Pinned speakers go to their
        configured slot, active sticky slots are reused, otherwise the lowest
        free slot is taken. Time boundaries are half-open ``[start, end)`` so
        a cue ending exactly when another starts frees its slot first.
        """
        pinned = self.speaker_slots or {}
        sticky = {}  # speaker -> slot currently associated with the speaker
        active = {}  # slot -> (end_time, cue_index)
        slots = [None] * len(cues)

        for index in ordered:
            cue = cues[index]
            speaker = self._resolve_speaker(cue)

            # Expire cues whose end time has been reached.
            for slot in list(active):
                if active[slot][0] <= cue.start:
                    del active[slot]

            chosen = None
            if speaker is not None and speaker in pinned:
                chosen = pinned[speaker]
                if chosen in active and active[chosen][1] != index:
                    raise SubtitleLayoutError(
                        "Speaker %r is pinned to slot %d but two of its cues "
                        "overlap at t=%ss." % (speaker, chosen, cue.start)
                    )
            if chosen is None:
                own_sticky = sticky.get(speaker) if speaker is not None else None
                candidates = [
                    slot
                    for slot in range(self.max_lines)
                    if slot not in active and slot not in pinned.values()
                ]
                if own_sticky is not None and own_sticky in candidates:
                    # Reclaim the speaker's own row as soon as it frees up,
                    # even if another speaker's lower empty row exists. This
                    # is what makes a speaker return home after an overlap.
                    chosen = own_sticky
                elif candidates:
                    chosen = candidates[0]
            if chosen is None:
                for slot in range(self.max_lines):
                    if slot not in active:
                        chosen = slot
                        break
            if chosen is None:
                raise SubtitleLayoutError(
                    "More than %d simultaneous subtitles at t=%ss; cannot place "
                    "cue %r. Increase max_lines or adjust the cues."
                    % (self.max_lines, cue.start, cue.text)
                )

            slots[index] = chosen
            active[chosen] = (cue.end, index)
            if speaker is not None and speaker not in pinned:
                sticky[speaker] = chosen

        return slots

    def _compute_positions(self, cues, sizes, slots):
        placements = [None] * len(cues)

        # Segment the timeline at every cue boundary.
        boundaries = sorted(
            {boundary for cue in cues for boundary in (cue.start, cue.end)}
        )

        # Cache the stacked rows per boundary: {boundary: {slot: top_y}}.
        rows_by_boundary = {}
        for boundary in boundaries:
            active_indices = [
                index
                for index, cue in enumerate(cues)
                if cue.start <= boundary < cue.end
            ]
            if active_indices:
                rows_by_boundary[boundary] = self._stack_rows(
                    cues, sizes, slots, active_indices, boundary
                )

        for index, cue in enumerate(cues):
            text_width, text_height = sizes[index]
            slot = slots[index]
            x = self._horizontal_position(text_width)
            segments = []
            for boundary in boundaries:
                if not (cue.start <= boundary < cue.end):
                    continue
                rows = rows_by_boundary[boundary]
                if slot not in rows:  # pragma: no cover - invariant guard
                    raise SubtitleLayoutError(
                        "Internal layout error: cue %r lost its slot at t=%ss."
                        % (cue.text, boundary)
                    )
                y = rows[slot]
                self._check_inside_frame(cue, text_width, text_height, x, y)
                segments.append((boundary, x, y))

            placements[index] = SubtitlePlacement(
                cue=cue,
                slot=slot,
                positions=tuple(segments),
                size=(text_width, text_height),
            )

        return placements

    def _stack_rows(self, cues, sizes, slots, active_indices, boundary):
        """Return ``{slot: top_y}`` for rows visible at ``boundary``.

        Rows are stacked from the bottom margin upward, using the tallest clip
        of each row for its height. Pinned slots always keep their row even
        when empty: a speaker pinned above the bottom row never slides down.
        """
        height = self.size[1]
        active = {}
        for index in active_indices:
            slot = slots[index]
            active[slot] = max(active.get(slot, 0), sizes[index][1])

        pinned_slots = set((self.speaker_slots or {}).values())
        # Every pinned slot reserves a row for the whole video, including when
        # its speaker is currently silent, so pinned rows never slide. This
        # method is only called for boundaries with at least one visible cue,
        # so ``active`` is never empty and a real row height is always known.
        row_range = max(max(active), max(pinned_slots, default=-1)) + 1
        gap_height = max(active.values())

        rows = {}
        cursor = height - self.bottom_margin
        for slot in range(row_range):
            if slot in active:
                row_height = active[slot]
            else:
                # An empty slot below an occupied one. This is either a
                # reserved pinned row or, with pinned slots configured, an
                # intervening gap: in both cases it keeps its row height so
                # higher rows never slide down.
                row_height = gap_height
            cursor -= row_height
            rows[slot] = cursor
            cursor -= self.line_spacing

        topmost_y = min(rows.values())
        if topmost_y < self.top_margin:
            offender = next(
                cues[idx].text for idx in active_indices if slots[idx] == max(active)
            )
            raise SubtitleLayoutError(
                "Not enough vertical space at t=%ss: subtitles need %dpx above "
                "the bottom margin but only %dpx are available (frame height "
                "%dpx, top margin %dpx). Failing cue: %r. Reduce font size, "
                "line_spacing, margins or max_lines."
                % (
                    boundary,
                    height - self.bottom_margin - topmost_y,
                    self.safe_height,
                    height,
                    self.top_margin,
                    offender,
                )
            )
        return rows

    def _horizontal_position(self, text_width):
        if self.align == "left":
            return self.side_margin
        if self.align == "right":
            return self.size[0] - self.side_margin - text_width
        return (self.size[0] - text_width) // 2

    def _check_inside_frame(self, cue, text_width, text_height, x, y):
        if x < self.side_margin or x + text_width > self.size[0] - self.side_margin:
            raise SubtitleLayoutError(
                "Cue %r does not fit horizontally at t=%ss (x=%d, width=%d)."
                % (cue.text, cue.start, x, text_width)
            )
        if y < self.top_margin or y + text_height > self.size[1] - self.bottom_margin:
            raise SubtitleLayoutError(
                "Cue %r does not fit vertically at t=%ss (y=%d, height=%d)."
                % (cue.text, cue.start, y, text_height)
            )


def cues_from_srt(filename, encoding=None, speaker_parser=None):
    """Read an SRT file into :class:`SubtitleCue` objects.

    The raw subtitle text is kept verbatim. Speaker detection only fills the
    ``speaker`` attribute and never changes ``text``.

    Parameters
    ----------
    filename : str or path-like
        Path to the ``.srt`` file.
    encoding : str or None
        Optional file encoding, forwarded to the underlying SRT parser.
    speaker_parser : callable or None
        Function ``text -> speaker``. Defaults to
        :func:`default_speaker_parser`; pass ``None`` to keep every speaker
        unset.
    """
    raw = file_to_subtitles(filename, encoding=encoding)
    parser = default_speaker_parser if speaker_parser is None else speaker_parser
    cues = []
    for (start, end), text in raw:
        if start is None or end is None:
            raise ValueError(
                "Malformed SRT entry %r: missing timing information." % text
            )
        cues.append(
            SubtitleCue(
                start=float(start),
                end=float(end),
                text=text,
                speaker=parser(text) if parser else None,
            )
        )
    return cues


def position_subtitle_clips(cues, make_textclip, layout):
    """Render and position text clips for a set of cues.

    Each cue is rendered exactly once through ``make_textclip``; the returned
    clips are already timed (``start``/``end``), positioned (a time-aware
    ``pos`` callable reflecting the layout segments) and layered by slot, so
    they can be passed straight to
    :class:`~moviepy.video.compositing.CompositeVideoClip.CompositeVideoClip`.

    Parameters
    ----------
    cues : iterable of SubtitleCue
        Cues to render.
    make_textclip : callable
        Function ``text -> VideoClip``, typically a ``TextClip`` factory such
        as the one returned by :func:`make_default_textclip`. The returned
        clip must expose a real pixel ``size``; its text must be the cue text
        rendered verbatim.
    layout : SubtitleLayout
        Layout configuration.

    Returns
    -------
    list of VideoClip
        Positioned clips in the cues' order.
    """
    cues = list(cues)
    clips = [make_textclip(cue.text) for cue in cues]

    text_sizes = {}
    for index, clip in enumerate(clips):
        if clip is None or not getattr(clip, "size", None):
            raise ValueError(
                "make_textclip returned no sized clip for cue %r." % cues[index].text
            )
        text_sizes[index] = tuple(int(value) for value in clip.size)

    placements = layout.layout(cues, text_sizes)

    positioned = []
    for clip, placement in zip(clips, placements):
        cue = placement.cue
        duration = cue.end - cue.start
        clip = clip.with_start(cue.start).with_duration(duration)
        clip = clip.with_position(
            lambda t, placement=placement: placement.position_at(
                placement.cue.start + max(0.0, t)
            )
        )
        clip = clip.with_layer_index(placement.slot)
        positioned.append(clip)
    return positioned


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)
