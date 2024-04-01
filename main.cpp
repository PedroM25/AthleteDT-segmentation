#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

const int IN_W = 300;
const int IN_H = 300;

const float CONF_TRSH = 0.9;
const float MEAN_SUBTRACTION_VAL = 127.5; // Result from doing 255/2
const float SCALING_FACTOR = 0.00784; // Result from doing 2/255
const float SECONDS_BETW_DETECTIONS = 3;

const std::string APP_NAME = "AthleteDT-seg";
const std::string WIN_NAME = "Output";
const std::string MODEL_WEIGHTS_PATH = "model/mask-rcnn-coco/frozen_inference_graph.pb";
const std::string MODEL_CONFIG_PATH = "model/mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
const std::string CLASS_NAMES_PATH = "model/mask-rcnn-coco/object_detection_classes_coco.txt";
std::vector<std::string> class_names{};

const std::vector<cv::Scalar> COLORS{
    cv::Scalar{0,255,0},
    cv::Scalar{0,0,255},
    cv::Scalar{255,0,0},
    cv::Scalar{0,255,255},
    cv::Scalar{255,255,0},
    cv::Scalar{255,0,255}
};

std::string current_timestamp(){
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S");
    return oss.str();
}

bool readClassNames(){
    std::ifstream file(CLASS_NAMES_PATH);
    class_names = std::vector<std::string>{};
    
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            class_names.push_back(line);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char **argv){
    if (argc != 2) {
        std::cout << "Usage: " << APP_NAME << " <video path>" << std::endl;
        return 1;
    }

    // read class names
    if (!readClassNames()){
        return 1;
    };
    
    // create output/ folder
    std::filesystem::create_directories("./output");
    
    std::string start_timestamp = current_timestamp();

    // video capture
    cv::VideoCapture cap{argv[1]};
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file. Exiting." << std::endl;
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS); // video framerate
    int delay = 1000 / fps; // Calculate delay based on the framerate
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    //int num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

    // video writer
    std::string video_file_name = "output/output_" + start_timestamp + ".avi";
    cv::VideoWriter output_video_writer = cv::VideoWriter(video_file_name, cv::VideoWriter::fourcc('H','2','6','4'), fps, cv::Size(frame_width,frame_height));
    
    // pre trained network
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(MODEL_WEIGHTS_PATH, MODEL_CONFIG_PATH);

    cv::Mat frame{};
    while (cap.read(frame)){
        
    }
    cap.release();
    cv::destroyAllWindows();
}
