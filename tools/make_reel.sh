#!/bin/bash
# Builds the reel PipeGen runs on: the five fixed-camera scenes back to back as
# one H.264 file, so a single pipeline run (FP16 or INT8) covers the whole
# benchmark. Run inside WSL after tools/fetch_clips.sh:
#   bash tools/make_reel.sh ~/videos/crosscount/raw ~/videos/crosscount
#
# 1280x720 at 30 fps: the detector sees a 640-pixel input either way, and 30 fps
# halves the work on the 60 fps sources with no loss for tracking at road speeds.
# A GAP_S black gap between scenes empties the picture, so tracks end at a scene
# change instead of jumping from one street to the next. The manifest it writes
# (frame range per scene) is how CrossCount knows which counting line applies.
# in_street.mp4 is downloaded but left out: the camera pans (tools/check_clips.py).
set -euo pipefail
SRC=${1:?raw clip folder}
OUT=${2:?output folder}
FPS=30 GAP_S=1 W=1280 H=720
SCENES=(in_citycentre nl_amsterdam wroclaw_junction tram_junction wroclaw_zebra)
mkdir -p "$OUT/scenes"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

ENC=(-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -bf 0 -g $FPS -r $FPS -an)
ffmpeg -v error -y -f lavfi -i "color=black:s=${W}x${H}:r=$FPS:d=$GAP_S" "${ENC[@]}" "$TMP/gap.mp4"

: > "$TMP/list.txt"
frame=0
{
  echo "{\"fps\": $FPS, \"width\": $W, \"height\": $H, \"scenes\": ["
  first=1
  for s in "${SCENES[@]}"; do
    ffmpeg -v error -y -i "$SRC/$s.mp4" -vf "scale=$W:$H,fps=$FPS" "${ENC[@]}" "$OUT/scenes/$s.mp4"
    n=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$OUT/scenes/$s.mp4")
    [ $first -eq 1 ] || echo ","
    first=0
    printf '  {"scene": "%s", "start_frame": %d, "end_frame": %d}' "$s" "$frame" "$((frame + n))"
    frame=$((frame + n + GAP_S * FPS))
    echo "file '$OUT/scenes/$s.mp4'" >> "$TMP/list.txt"
    echo "file '$TMP/gap.mp4'" >> "$TMP/list.txt"
  done
  printf '\n]}\n'
} > "$OUT/crosscount_reel.json"

ffmpeg -v error -y -f concat -safe 0 -i "$TMP/list.txt" "${ENC[@]}" "$OUT/crosscount_reel.mp4"
got=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$OUT/crosscount_reel.mp4")
echo "reel: $OUT/crosscount_reel.mp4  frames=$got expected=$frame"
cat "$OUT/crosscount_reel.json"
