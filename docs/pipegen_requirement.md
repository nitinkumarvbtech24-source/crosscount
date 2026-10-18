# PipeGen requirement (Stage 1: Requirement)

Paste only the block below as the opening message of a new PipeGen 0.5.1 project:

```
Traffic and crosswalk analytics for a pole-mounted city camera.
Input: one video file, /home/dhyey/videos/crosscount/crosscount_reel.mp4 (1280x720, 30 fps): five fixed street cameras back to back. This is the only input.
Detect pedestrians (person), bicycles, motorbikes, cars, buses and trucks, and track each with a persistent ID.
Publish every frame's objects as JSON over MQTT to localhost:1883, topic crosscount/detections: the frame number and timestamp, and for each object the tracker ID, class, confidence and bounding box.
Also stream the annotated video over RTSP to rtsp://localhost:8554/crosscount and save it to a file.
Privacy: before any frame is displayed, saved or streamed, pixelate every person's head region.
Compile the detector to TensorRT in FP16 and INT8 and report speed, power and accuracy for both.
```

Answers for the cards PipeGen shows next:

| Card | Answer | Why |
|---|---|---|
| Privacy: head or face? | Entire head region | the detector finds people, not faces |
| Inputs | **1** input: the reel | |
| Outputs | MQTT `localhost:1883` topic `crosscount/detections`; RTSP `rtsp://localhost:8554/crosscount`; Saved video | |
| Hide before saving or sending? | Faces | becomes head blur, as no task finds faces |
| Does this camera move? | Not sure | "Fixed" enables frame skipping and the simple tracker; counting needs every frame |
| How is it mounted? | Angled on a wall | keeps the stronger tracker |
| How big are the targets? | Small or far | no frame skipping |
| Model | a COCO detector (YOLO11/YOLOv8 class): it has person, bicycle, car, motorcycle, bus, truck | |
| Compiler | FP16 first, then INT8 | |

Why the first attempt was redone: an earlier version of this text said "later the same pipeline runs on a live RTSP camera" and asked for number-plate pixelation. PipeGen turned the first into a second input (pointed at the pipeline's own RTSP output, which would loop) and the second into an extra plate-detection model. Neither is needed to show the system; a live camera is a change of source in the generated run.sh.

Counting lines are not set in PipeGen: CrossCount applies a different line per
scene, and the reel changes scene four times. PipeGen does all the perception
(detector on TensorRT, tracker, privacy, MQTT); CrossCount turns its tracks into
counts.

Evidence to screenshot for the submission:
1. the Pipeline IR node graph,
2. the TensorRT speed / power / accuracy report for FP16 and for INT8,
3. the Run stage with the pipeline live.
