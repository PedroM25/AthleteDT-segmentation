#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cout << "usage: PedroSwissTiming <video path>" << std::endl;
        return -1;
    }

    cv::VideoCapture cap{argv[1]};
    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS); // Get the framerate of the video
    int delay = 1000 / fps; // Calculate delay based on the framerate

    cv::HOGDescriptor hog{};
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    cv::Mat frame{};
    while (cap.read(frame)){

        // TODO
        std::vector<cv::Rect> found{};
        hog.detectMultiScale(frame,found, 0, cv::Size(8,8), cv::Size(32,32));

        for (auto rec : found){
            
            cv::rectangle(frame, rec.tl(), rec.br(), cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Original video", frame);

        if (cv::waitKey(delay) == 27) { //ESC key
            break;
        }
    }

    std::cout << "No more frames grabbed. Exiting..." << std::endl;
    cap.release();
    cv::destroyAllWindows();
}