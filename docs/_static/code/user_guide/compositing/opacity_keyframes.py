"""In this example, a coloured caption fades in, holds, then fades out over a
background video.

Keyframe times are relative to the beginning of the caption clip, so they keep
working when the caption is moved with ``with_start``. Opacity is interpolated
at the exact time each frame is sampled, even when a fade boundary falls
between two frames of the output.
"""

from moviepy import ColorClip, CompositeVideoClip, VideoFileClip, vfx

background = VideoFileClip("example.mp4")

# A caption that fades in over 0.5s, holds for 2.5s, then fades out over 0.5s
caption = ColorClip((400, 120), color=(255, 255, 255), duration=4)
caption = caption.with_effects(
    [
        vfx.OpacityKeyframes(
            [
                (0.0, 0.0),  # transparent at the start
                (0.5, 1.0),  # fully visible after half a second
                (3.0, 1.0),  # still visible until 3s
                (3.5, 0.0),  # transparent again at 3.5s
            ]
        )
    ]
)

final_clip = CompositeVideoClip(
    [background, caption.with_position("center").with_start(2)]
)
final_clip.write_videofile("opacity_keyframes.mp4")
