My attempt at the Swiss Timing coding challenge

## Solution

This solution uses the Mask-RCNN pre-trained model from TensorFlow for object detection. Downloaded from [TensorFlow](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz) and the repo for the implementation of Mask-RCNN can be seen [here](https://github.com/matterport/Mask_RCNN)
* Confidence considered for object detection model: 0.8
* Libraries used: OpenCV 4.9.0, built from source
* Output stored in a `output/` under same folder as bin

This solution performs object detection in every frame. Only the first object classified by the model as a "person", with a confidence above 0.8 is considered.

Any other "person" detected is ignored.

## Usage

`AthleteDT-seg <video path> [--visualize]`

The `--visualize` flag is boolean and if passed as an argument allows you to show the frame post-process live.

## Example output 1

**Command:**

`AthleteDT-seg input_video/ice_skating2_4s.mp4`

**Video:**

[output_ice_skating2_4s.webm](https://github.com/PedroM25/AthleteDT-segmentation/assets/40021588/96b36d1d-8ce9-4224-9dba-40e1cb2560b3)

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
FRAME 7: Person detected, confidence: 0.999276, coordinates: [[1033,324],[1422,324],[1033,928],[1422,928]]
... snip ...
FRAME 111: Person detected, confidence: 0.996296, coordinates: [[577,225],[948,225],[577,731],[948,731]]
FRAME 112: Person detected, confidence: 0.997915, coordinates: [[581,223],[940,223],[581,728],[940,728]]
FRAME 113: Person detected, confidence: 0.996441, coordinates: [[603,191],[934,191],[603,731],[934,731]]
FRAME 114: Person detected, confidence: 0.99737, coordinates: [[607,214],[935,214],[607,739],[935,739]]
FRAME 115: Person detected, confidence: 0.993023, coordinates: [[614,217],[942,217],[614,745],[942,745]]
No more frames grabbed. Exiting...
Total number of frames processed: 115
Processing time: 246.602s
```

## Example output 2

**Command:**

`AthleteDT-seg input_video/skate1_4s.mp4`

**Video:**

[output_skate1_4s.webm](https://github.com/PedroM25/AthleteDT-segmentation/assets/40021588/94aef04e-30cf-4463-a8cb-fb05b8a2c393)

**Log file:**

```log
Starting AthleteDT-seg execution.
Successfully imported video "skate1_4s.mp4", FPS: 25.1751, Num frames: 117, Resolution: 1280x720
FRAME 1: Person detected, confidence: 0.997617, coordinates: [[469,150],[765,150],[469,426],[765,426]]
FRAME 2: Person detected, confidence: 0.998832, coordinates: [[481,152],[778,152],[481,430],[778,430]]
FRAME 3: Person detected, confidence: 0.99859, coordinates: [[486,164],[775,164],[486,425],[775,425]]
FRAME 4: Person detected, confidence: 0.998743, coordinates: [[492,162],[763,162],[492,435],[763,435]]
FRAME 5: Person detected, confidence: 0.998116, coordinates: [[515,175],[764,175],[515,453],[764,453]]
... snip ...
FRAME 114: Person detected, confidence: 0.992107, coordinates: [[345,231],[674,231],[345,480],[674,480]]
FRAME 115: Person detected, confidence: 0.992937, coordinates: [[313,238],[686,238],[313,497],[686,497]]
FRAME 116: Person detected, confidence: 0.994564, coordinates: [[324,249],[686,249],[324,512],[686,512]]
FRAME 117: Person detected, confidence: 0.974667, coordinates: [[309,253],[692,253],[309,525],[692,525]]
No more frames grabbed. Exiting...
Total number of frames processed: 117
Processing time: 187.847s
```
