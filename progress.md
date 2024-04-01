# My progress:

## Friday, 29th of March

I figured that there are 3 strategies to perform object detection:

My attempt at the Swiss Timing exercise

3 strategies used for object detection:

- deep learning object detection model
- Haar Cascade classifier
- Histogram of Oriented Gradients

Was using libopencv-dev but that was version 4.5
Compiled from source and I am now using 4.9.0

One of the main problems was getting OpenCV working in the first place but that is solved

## Saturday, 30th of March

I don't want to train my own models so I searched for some pre-trained ones.
I was able to find some but very little information on scale, mean subtraction and channels order for those pretrained models
    models to use:  https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
    some page with info I found: https://github.com/LUOBO123LUOBO123/Mobilenet-ssd-v2/tree/master/dnn

TensorFlow models from Kaggle are not really possible to use?
I have the .pb but it seems I need to freeze the models?
Also, no .pbtxt files for those
They didn't work out the box

I hear about something called cppflow to try to use TensorFlow withing C++ but scrapped it

Also, the following models seem to work with no hiccups:

// Load pre-trained SSD model
    
//WORKED
//cv::dnn::Net net = cv::dnn::readNetFromTensorflow("model/Faster-RCNNInceptionV2/frozen_inference_graph.pb", "model/Faster-RCNNInceptionV2/config.pbtxt");
//cv::dnn::Net net = cv::dnn::readNetFromTensorflow("model/MobileNet-SSDv1/frozen_inference_graph.pb", "model/MobileNet-SSDv1/config.pbtxt");
//cv::dnn::Net net = cv::dnn::readNetFromTensorflow("model/MobileNet-SSDv2/frozen_inference_graph.pb", "model/MobileNet-SSDv2/config.pbtxt");
//cv::dnn::Net net = cv::dnn::readNetFromTensorflow("model/MobileNet-SSDv3/frozen_inference_graph.pb");
//cv::dnn::Net net = cv::dnn::readNetFromTensorflow("model/MobileNet-SSDv3/frozen_inference_graph.pb", "model/MobileNet-SSDv3/config.pbtxt");
//cv::dnn::Net net = cv::dnn::readNetFromCaffe("model/MobileNet-SSDCaffe/MobileNetSSD_deploy.prototxt",
                "model/MobileNet-SSDCaffe/MobileNetSSD_deploy.caffemodel");


Ok I now realize that I just need to detect the athlete in the first frame and THEN apply a tracking algorithm to the rest of video, based on the box of the first detection


## Sunday, 31st of March

single biggest help getting the pretrained model to work:
https://pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

writing frames to video:
https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

great example that helped me understand how everything worked:
https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp

great tutorial for the tracking and explanation of tracking and why you need it:
https://learnopencv.com/object-tracking-using-opencv-cpp-python/


## Monday, 1st of April

https://ftp.up.pt/kde-applicationdata/kdenlive/motion-tracker/DaSiamRPN/


background removal for object detection:
https://www.youtube.com/watch?v=O3b8lVF93jU

