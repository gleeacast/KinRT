# Real-Robot Media Record

The project authors supplied the source media in two external archives. The
archives are not included in this repository. This record documents the
derivatives published on the KinRT project page.

## Video Derivatives

| Published file | Source entry | Source interval | Published format |
| --- | --- | --- | --- |
| `videos/diyrobot-handover.mp4` | `handover1.mp4` | 0.5-13.5 s | H.264, 1280 x 720, 24 FPS |
| `videos/diyrobot-press-button.mp4` | `press_button1.mp4` | 0.2-10.2 s | H.264, 1280 x 720, 24 FPS |
| `videos/diyrobot-rotate-screwdriver.mp4` | `rotate2.mp4` | 1.0-15.5 s | H.264, 1280 x 720, 24 FPS |

The clips preserve real-time playback. Transcoding drops frames from 30 to 24
FPS, removes audio, uses YUV 4:2:0 pixel format, and places MP4 metadata at the
start of each file for progressive web playback. No motion segment was sped up
or slowed down.

## Still Images

The five task images under `images/diyrobot/` are resized JPEG derivatives of
the corresponding keyframes in `frames.zip`. Each image is center-cropped to
960 x 540 for a consistent 16:9 presentation. The three video posters are
single-frame derivatives of their respective source videos.

The published media is presentation material. It does not replace trial-level
evaluation logs or constitute additional quantitative evidence.
