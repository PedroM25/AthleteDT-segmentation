#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

namespace fs = std::filesystem;

const float CONF_TRSH = 0.8;
const float MASK_TRSH = 0.3;

const std::string APP_NAME = "AthleteDT-seg";
const std::string WIN_NAME = "Output";
const std::string MODEL_WEIGHTS_PATH = "model/mask-rcnn-coco/frozen_inference_graph.pb";
const std::string MODEL_CONFIG_PATH = "model/mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
const std::string CLASS_NAMES_PATH = "model/mask-rcnn-coco/object_detection_classes_coco.txt";

const cv::Scalar COLOR{0,255,0};

std::vector<std::string> class_names{};
std::ofstream log_file;
int frame_count = 0;

/*
 * Read from file supported objects that model can identify
 */
bool readClassNames(){
    std::ifstream file(CLASS_NAMES_PATH);
    
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

void drawMaskAndBBox(cv::Mat& frame, int classId, float conf, cv::Rect& bbox, cv::Mat& objectMask){
    // bounding box
    rectangle(frame, cv::Point(bbox.x, bbox.y), cv::Point(bbox.x+bbox.width, bbox.y+bbox.height), COLOR, 3);

    // label
    std::string label = class_names[classId] + " : " + std::to_string(conf);
    putText(frame, label, cv::Point(bbox.x, bbox.y - 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 1.5);

    // preprare mask: resize, threshold, color
    cv::resize(objectMask, objectMask, cv::Size(bbox.width, bbox.height));
    cv::Mat mask = objectMask > MASK_TRSH;
    cv::Mat coloredRoi = 0.3 * COLOR + 0.7 * frame(bbox);
    coloredRoi.convertTo(coloredRoi, CV_8UC3);

    // prepare contours
    std::vector<cv::Mat> contours;
    cv::Mat hierarchy;
    mask.convertTo(mask, CV_8U);
    cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(coloredRoi, contours, -1, COLOR, 5, cv::LINE_8, hierarchy, 100);

    // apply
    coloredRoi.copyTo(frame(bbox), mask);
}

void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs){
    cv::Mat outDetections = outs[0];
    cv::Mat outMasks = outs[1];

    int numDetections = outDetections.size[2];

    outDetections = outDetections.reshape(1, outDetections.total() / 7);
    for (int i = 0; i < numDetections; i++){
        float confidence = outDetections.at<float>(i, 2);
        int classId = outDetections.at<float>(i, 1);

        if (confidence > CONF_TRSH && class_names[classId] == "person"){
            // Extract the bounding box
            int xLeftTop = outDetections.at<float>(i, 3) * frame.cols;
            int yLeftTop =  outDetections.at<float>(i, 4) * frame.rows;
            int xRightBottom = outDetections.at<float>(i, 5) * frame.cols;
            int yRightBottom = outDetections.at<float>(i, 6) * frame.rows;

            log_file << "FRAME " << frame_count << ": " << "Person detected, confidence: " << confidence << ", coordinates: [" 
                << "["<<xLeftTop<<","<<yLeftTop<<"],"
                << "["<<xRightBottom<<","<<yLeftTop<<"],"
                << "["<<xLeftTop<<","<<yRightBottom<<"],"
                << "["<<xRightBottom<<","<<yRightBottom<<"]"
                << "]"
                << std::endl;

            // keep boxes inside of bounds
            xLeftTop = std::max(0, std::min(xLeftTop, frame.cols -1));
            yLeftTop = std::max(0, std::min(yLeftTop, frame.rows -1));
            xRightBottom = std::max(0, std::min(xRightBottom, frame.cols -1));
            yRightBottom = std::max(0, std::min(yRightBottom, frame.rows -1));
            
            cv::Rect box{xLeftTop, yLeftTop, xRightBottom - xLeftTop + 1, yRightBottom - yLeftTop + 1};

            // extract mask
            cv::Mat objectMask{outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i,classId)};

            drawMaskAndBBox(frame, classId, confidence, box, objectMask);
            break; //first person detected only; others are ignored
        }
    }
}

int main(int argc, char **argv){
    
    if (argc == 1) {
        std::cerr << "No arguments passed. Usage: " << APP_NAME << " <video path> [--visualize]" << std::endl;
        return 1;
    }

    // check if should show live processing of frames
    bool showProcessing = false;
    if (argc == 3 && std::string(argv[2]) == "--visualize"){
        showProcessing = true;
    }

    if (!readClassNames()){
        return 1;
    };

    // create output/ folder
    fs::create_directories("./output");
    
    // Create log file
    std::string video_file_name = fs::path(argv[1]).stem();
    std::string log_file_name = "output/" + APP_NAME + "_" + video_file_name + ".log";
    log_file= std::ofstream(log_file_name, std::ios::trunc);
    if (!log_file.is_open()) {
        std::cerr << "Error creating log file. Exiting." << std::endl;
        return 1;
    }

    log_file << "Starting " << APP_NAME << " execution." << std::endl;

    // video capture
    cv::VideoCapture cap{argv[1]};
    if (!cap.isOpened()) {
        log_file << "Error opening video stream or file. Exiting." << std::endl;
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    log_file << "Successfully imported video " << fs::path(argv[1]).filename()
        << ", FPS: " << fps 
        << ", Num frames: " << num_frames 
        << ", Resolution: " << frame_width << "x" << frame_height
        << std::endl;

    // video writer
    std::string output_file_name = "output/output_" + video_file_name + ".mp4";
    cv::VideoWriter output_video_writer = cv::VideoWriter(output_file_name, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(frame_width,frame_height));
    
    // pre trained network
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(MODEL_WEIGHTS_PATH, MODEL_CONFIG_PATH);

    int delay = 1000 / fps; // Calculate delay based on the framerate
    int64_t start = cv::getTickCount();
    std::cout << "Video processing has started..." << std::endl;
    cv::Mat frame{};
    while (cap.read(frame)){
        frame_count++;
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1., cv::Size(frame.cols, frame.rows), cv::Scalar(), true, false);
        net.setInput(blob);

        std::vector<std::string> out_names{"detection_out_final", "detection_masks"};
        std::vector<cv::Mat> outs{};
        net.forward(outs, out_names);

        postprocess(frame, outs);

        output_video_writer.write(frame);
        if(showProcessing){
            cv::imshow(WIN_NAME, frame);
        }

        if (cv::waitKey(delay) == 27) { //ESC key
            break;
        }
    }
    std::cout << "Processing finished. Check output folder" << std::endl;

    log_file << "No more frames grabbed. Exiting..." << std::endl;
    log_file << "Total number of frames processed: " << frame_count << std::endl;
    int64_t end = cv::getTickCount();
    log_file << "Processing time: " << (end-start)/cv::getTickFrequency() << "s" << std::endl;
    cap.release();
    cv::destroyAllWindows();
    log_file.close();
}
