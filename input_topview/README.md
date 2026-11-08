Put the nadir capture here:

- `topview.srt` (required): a mini DJI SRT naming the nadir frame (`FrameCnt`) and the drone
  position/gimbal at that frame. If it holds several blocks, the one with pitch closest to -90 is used.
- optional nadir image (`.jpg/.jpeg/.png/.tif`): used as the frame's pixels. Without an image,
  the frame is read from `inputs/flight.mp4` at that FrameCnt.

Polygon segmentation runs on this frame. Contents are git-ignored.
