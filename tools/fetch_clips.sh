#!/bin/bash
# Downloads the benchmark footage: six fixed-camera street crossings from
# Pexels (free licence, no attribution required), 1080p. Run inside WSL:
#   bash tools/fetch_clips.sh ~/videos/crosscount/raw
# PipeGen's container only sees videos under /home, so keep them there.
set -euo pipefail
OUT=${1:?output folder}
mkdir -p "$OUT"
while read -r name url page; do
  [ -z "$name" ] && continue
  if [ -s "$OUT/$name.mp4" ]; then echo "have $name"; continue; fi
  echo "get  $name  <- $page"
  curl -sSfL --retry 3 -o "$OUT/$name.mp4.part" "$url"
  mv "$OUT/$name.mp4.part" "$OUT/$name.mp4"
done <<'EOF'
in_citycentre  https://videos.pexels.com/video-files/28925595/12518849_1920_1080_60fps.mp4  https://www.pexels.com/video/28925595/
in_street      https://videos.pexels.com/video-files/30850558/13193034_1920_1080_30fps.mp4  https://www.pexels.com/video/30850558/
wroclaw_junction  https://videos.pexels.com/video-files/33865601/14371841_1920_1080_60fps.mp4  https://www.pexels.com/video/33865601/
nl_amsterdam   https://videos.pexels.com/video-files/12545270/12545270-hd_1920_1080_60fps.mp4 https://www.pexels.com/video/12545270/
tram_junction    https://videos.pexels.com/video-files/40260363/17195778_1920_1080_60fps.mp4  https://www.pexels.com/video/40260363/
wroclaw_zebra    https://videos.pexels.com/video-files/40260120/17195648_1920_1080_60fps.mp4  https://www.pexels.com/video/40260120/
EOF
ls -la "$OUT"
