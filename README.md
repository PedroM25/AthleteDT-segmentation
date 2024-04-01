My attempt at the Swiss Timing coding challenge

## Summary
This solution uses the OpenCV library 4.9.0, the MobileNet SSD Caffe pretrained model from [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) for object detection and the CSRT tracker provided in OpenCV framework for object tracking.

In my search I discovered several ways to perform object detection:
* Histogram of gradients (HOG)
* Haar Cascades
* Deep Learning-based object detection using SSD, YOLO, Faster R-CNN

For object tracking there are also several algorithms provided by OpenCV out-of-the-box:
* CSRT
* DaSiamRPN
* GOTURN
* KCF
* MIL
* Nano
* Vit

Source: https://docs.opencv.org/4.9.0/d0/d0a/classcv_1_1Tracker.html

During setup, I had to compile OpenCV from source as the libopencv-dev Ubuntu package was outdated (4.5.5 version).

### Object detection

I decided to use a Deep Learning-based object detection method because it was the most reliable of all methods considered.
Unlike HOG, which produced a lot of false-positives and was not accurate at all, the DL model was able to accurately detect a person in the first few frames of any of clips used.

Due to time constraints I used a pretrained model, the Caffe implementation of MobileNet SSD.

Indeed, I was successful in importing other pretrained models from this ["opencv wiki page"](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API) but had a hard time finding scaling factors and mean subtraction values associated with them. This made image preprocessing more difficult.

### Object tracking
I used the CSRT tracking algorithm because it was the most reliable and easiest to apply, out of the box, without needing additional parameters.
I tried to use DaSiamRPN but realized I needed to pass in some additional parameters.

## Features requested
The following are the features requested and the progress of implementation of each:

|Feature|Progress|Info|
|---|---|---|
|Choose a sports clip (3-10 seconds)|✅|Tried solution with three different sports clips depicting fencing, skiing, figure skating|
|Load and process all images|✅|
|Suitable representation of the tracking results shall be visible in the output |✅|Bounding box|
|Further information can be extracted from the video and displayed (e.g. segmentation, skeleton recognition, ...)|❌|
|Information should be available in text format|✅|.log file created after each run|
|Visualized output should be available for playback after processing|✅|.avi file with H.264 codec exported after each run|
|Display output during processing|✅|


## Future work
Since the clips are short, this solution works fine as it is but tracking for long periods of time can lead to loss of tracking of the original object. To mitigate this, I would perform ocasional object detection and ensure that the program keeps tracking the desired object (the athlete).

I would also explore other tracking algorithms, other pretrained models for object detection and dedicate some time to understand what is out there and learning the math details behind the different approaches (SSD, Faster R-CNN, YOLO and others)

## Usage
* Build using cmake
* Execute PedroAthleteDT binary, passing "relative path of video to be analyzed" as argument

## Showcase

Executing the command `PedroAthleteDT input_video/fencing1.mp4` we get:

Output video:

[output.webm](https://github.com/PedroM25/SwissTimingExercise/assets/40021588/60b10b54-04a5-4628-a427-87cf89c82bcb)


Log produced:

```log
Starting PedroAthleteDT execution.
Successfully imported video. Video path: input_video/fencing1.mp4, FPS: 30, Resolution: 1280x720
FRAME 1: Person detected, confidence: 0.998385, coordinates: [[897,136],[1245,136],[897,523],[1245,523]]
FRAME 1: Target tracked: [[898,137],[1246,137],[898,524],[1246,524]]
FRAME 2: Target tracked: [[899,136],[1247,136],[899,523],[1247,523]]
FRAME 3: Target tracked: [[911,135],[1259,135],[911,522],[1259,522]]
FRAME 4: Target tracked: [[923,133],[1271,133],[923,520],[1271,520]]
... snip ...
FRAME 275: Target tracked: [[730,175],[994,175],[730,468],[994,468]]
FRAME 276: Target tracked: [[738,173],[1002,173],[738,466],[1002,466]]
FRAME 277: Target tracked: [[749,176],[1008,176],[749,464],[1008,464]]
FRAME 278: Target tracked: [[750,174],[1009,174],[750,462],[1009,462]]
FRAME 279: Target tracked: [[755,177],[1014,177],[755,465],[1014,465]]
FRAME 280: Target tracked: [[757,177],[1021,177],[757,470],[1021,470]]
No more frames grabbed. Exiting...
Total number of frames processed: 280
Processing time: 29.6899s

```
