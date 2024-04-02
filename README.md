My attempt at the Swiss Timing coding challenge

## Solution

This solution uses the Mask-RCNN pre-trained model from TensorFlow [(download here)](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz) for object detection.
* Confidence considered for object detection model: 0.6
* Libraries used: OpenCV 4.9.0, built from source
* Output stored in a `output/` under same folder as bin

This solution performs object detection in every frame. Only the first object classified by the model as a "person", with a confidence above 0.8 is considered.

Any other "person" detected is ignored.

## Usage

AthleteDT-seg <video path> [--visualize]

The `--visualize` flag is boolean and if passed as an argument allows you to show the frame post-process live.

## Example output

**Command:**

`AthleteDT-seg ../input_video/ice_skating2_4s.mp4`

**Video:**

[ice_skating2_4s.mp4]()

**Log file:**

```log
Starting AthleteDT-seg execution.
Successfully imported video "ice_skating2_4s.mp4", FPS: 25.1646, Num frames: 115, Resolution: 1920x1080
FRAME 1: Person detected, confidence: 0.995384, coordinates: [[1119,371],[1520,371],[1119,946],[1520,946]]
FRAME 2: Person detected, confidence: 0.988087, coordinates: [[1117,360],[1515,360],[1117,925],[1515,925]]
FRAME 3: Person detected, confidence: 0.996128, coordinates: [[1114,361],[1518,361],[1114,935],[1518,935]]
FRAME 4: Person detected, confidence: 0.998368, coordinates: [[1071,352],[1495,352],[1071,927],[1495,927]]
FRAME 5: Person detected, confidence: 0.999105, coordinates: [[1065,333],[1479,333],[1065,933],[1479,933]]
FRAME 6: Person detected, confidence: 0.999454, coordinates: [[1052,329],[1449,329],[1052,959],[1449,959]]
... snip ...
FRAME 113: Person detected, confidence: 0.996441, coordinates: [[603,191],[934,191],[603,731],[934,731]]
FRAME 114: Person detected, confidence: 0.99737, coordinates: [[607,214],[935,214],[607,739],[935,739]]
FRAME 115: Person detected, confidence: 0.993023, coordinates: [[614,217],[942,217],[614,745],[942,745]]
No more frames grabbed. Exiting...
Total number of frames processed: 115
Processing time: 243.522s

```
