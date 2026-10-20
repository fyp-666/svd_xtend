# Big Buck Bunny endpoint example

These two images are derived from *Big Buck Bunny* (2008), an open movie by the Blender Foundation.

- Official project and license statement: https://peach.blender.org/about/
- Official download page: https://peach.blender.org/download/
- Downloaded archive: https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_640x360.m4v.zip
- License: Creative Commons Attribution 3.0 Unported (CC BY 3.0), https://creativecommons.org/licenses/by/3.0/
- Required film-excerpt credit: (c) copyright 2008, Blender Foundation / www.bigbuckbunny.org
- Changes: video-frame extraction, RGB conversion, a centered crop, and Lanczos resizing; no audio is included.

## Sampling and preprocessing

| Image | Source timestamp | Decoded source | Saved image |
| --- | --- | --- | --- |
| first.png | 00:50.000 | 640 x 359 | 512 x 320 RGB PNG |
| last.png | 00:50.500 | 640 x 359 | 512 x 320 RGB PNG |

The official filename says 640x360; this file actually decodes as 640 x 359 pixels at 24 fps. FFmpeg accurate seeking selects the frame at each requested time. The timestamps are 0.5 seconds apart (12 source-frame intervals). Both show the same shot while the bunny stretches its arms.

Pillow `ImageOps.fit` uses a centered 574.4 x 359 crop (source box approximately x=32.8..607.2, y=0..359) and resizes it to 512 x 320 with Lanczos. This preserves geometry and removes narrow strips from the left and right. The original video remains in the ignored `tmp/demo_bunny_source/` cache.

Only the two input endpoints are included. No ground-truth intermediate frames are included, and this pair is a qualitative demonstration, not a benchmark. Model-generated frames are not frames from the original film. A generated sequence length, conditioning FPS, and export FPS must be recorded independently of this source interval.

## Reproduce

From the project root, in the dedicated Conda environment:

```powershell
conda run -n svd_interp python -m tools.prepare_demo --download
```

The script uses the bundled executable from imageio-ffmpeg and Pillow. It reuses the verified archive, writes only its fixed project paths, and refuses to replace a differing existing endpoint or this source note. Decoder or Pillow versions can affect PNG bytes.

Prepared with imageio-ffmpeg 0.6.0, Pillow 11.3.0, and FFmpeg `7.1-essentials_build-www.gyan.dev built with gcc 14.2.0 (Rev1, Built by MSYS2 project)`.

Archive SHA-256: `7118242b6728d40c871479c5b3c0f0fb27d748089df15d7f1b469f297c74a2d6`

Video SHA-256: `738e2f999860553d056dd79c952f58f63cbb73892a57c72342ce9e5330d9d2d7`

Endpoint SHA-256:

- `first.png`: `de665e164e830675c22f1db6ccbbfe8e1969a0e8c715f2a8722b7271c7e768f3`
- `last.png`: `3adf4a82a340c487129d4ce8bbfcaa19c0e9529f48af568eda3d5b57e4a8aab1`
