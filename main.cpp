#include <iostream>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

const int IN_W = 300;
const int IN_H = 300;

const float CONF_TRSH = 0.9;
const float MEAN_SUBTRACTION_VAL = 127.5; // Result from doing 255/2
const float SCALING_FACTOR = 0.00784; // Result from doing 2/255
const float SECONDS_BETW_DETECTIONS = 3;

const std::string APP_NAME = "PedroAthleteDT";
const std::string WIN_NAME = "Output";
const std::string PROTO_TXT_PATH = "model/MobileNet-SSDCaffe/MobileNetSSD_deploy.prototxt";
const std::string CAFFE_MODEL_PATH = "model/MobileNet-SSDCaffe/MobileNetSSD_deploy.caffemodel";
const std::vector<std::string> CLASS_NAMES{"background", "aeroplane", "bicycle", "bird", "boat", "bottle",
                                            "bus", "car", "cat", "chair", "cow", "diningtable",
                                            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                            "sofa", "train", "tvmonitor"};

std::ofstream log_file;
cv::VideoWriter output_video_writer;
int frame_count = 0;

std::string current_timestamp(){
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S");
    return oss.str();
}

/*
 * Returns true if a person is detected; false otherwise
 */
bool person_detected(cv::dnn::Net& net, const cv::Mat& frame, cv::Ptr<cv::Tracker>& tracker){
    bool success = false;
    cv::Mat blob = cv::dnn::blobFromImage(frame, SCALING_FACTOR, cv::Size(IN_W, IN_H), MEAN_SUBTRACTION_VAL);
    net.setInput(blob);
    cv::Mat detections = net.forward();

    cv::Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
    for (int i = 0; i < detectionMat.rows; i++){
        float confidence = detectionMat.at<float>(i, 2);
        int class_id = detectionMat.at<float>(i, 1);

        std::string label = CLASS_NAMES[class_id] + " : " + std::to_string(confidence);
        std::cout << "[DEBUG] "<< "Detected " << label << std::endl;

        if (CLASS_NAMES[class_id] == "person" && confidence > CONF_TRSH){
            success = true;

            // Object location
            int xLeftTop = detectionMat.at<float>(i, 3) * frame.cols;
            int yLeftTop = detectionMat.at<float>(i, 4) * frame.rows;
            int xRightBottom = detectionMat.at<float>(i, 5) * frame.cols;
            int yRightBottom = detectionMat.at<float>(i, 6) * frame.rows;
            
            log_file << "FRAME " << frame_count << ": " << "Person detected, confidence: " << confidence << ", coordinates: [" 
                << "["<<xLeftTop<<","<<yLeftTop<<"],"
                << "["<<xRightBottom<<","<<yLeftTop<<"],"
                << "["<<xLeftTop<<","<<yRightBottom<<"],"
                << "["<<xRightBottom<<","<<yRightBottom<<"]"
                << "]"
                << std::endl;

            cv::Point leftTopCoords{xLeftTop, yLeftTop};
            cv::Point rightBottomCoords{xRightBottom, yRightBottom};
            cv::Rect rec{leftTopCoords, rightBottomCoords};
            // draw rectangle
            cv::rectangle(frame, rec, cv::Scalar(0, 255, 0), 2);
            
            // init tracker
            tracker->init(frame, rec);

            // Draw label and confidence of prediction in frame
            cv::putText(frame, label, cv::Point(xLeftTop, yLeftTop - 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0),1.5);
            
            output_video_writer.write(frame);
            cv::imshow(WIN_NAME, frame);
            cv::waitKey(1600);
            break; //first one detected, others ignored
        }
    }
    return success;
}

int main(int argc, char **argv)
{
    int64_t start = cv::getTickCount();
    if (argc != 2) {
        std::cout << "Usage: " << APP_NAME << " <video path>" << std::endl;
        return 1;
    }
    
    // Create log file
    std::string start_timestamp = current_timestamp();
    std::string log_file_name = APP_NAME + "_" + start_timestamp + ".log";
    log_file= std::ofstream(log_file_name, std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Error creating log file." << std::endl;
        return 1;
    }

    log_file << "Starting " << APP_NAME << " execution." << std::endl;

    cv::VideoCapture cap{argv[1]};
    if (!cap.isOpened()) {
        log_file << "Error opening video stream or file. Exiting." << std::endl;
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS); // video framerate
    int delay = 1000 / fps; // Calculate delay based on the framerate
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    log_file << "Successfully imported video. "
        << "Video path: " << argv[1] 
        << ", FPS: " << fps 
        << ", Resolution: " << frame_width << "x" << frame_height
        << std::endl;

    // video output
    std::string video_file_name = "output_" + start_timestamp + ".avi";
    output_video_writer = cv::VideoWriter(video_file_name, cv::VideoWriter::fourcc('H','2','6','4'), fps, cv::Size(frame_width,frame_height));
    
    // pre trained network
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(PROTO_TXT_PATH, CAFFE_MODEL_PATH);

    // tracker
    cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();

    bool person_detected_first_time = false;
    cv::Mat frame{};
    while (cap.read(frame)){
        frame_count++;
        
        // 1. Detect a person 
        // If model detects more than one person, first detected will be considered
        if (!person_detected_first_time){
            person_detected_first_time = person_detected(net, frame, tracker);
            if(!person_detected_first_time){
                log_file << "FRAME " << frame_count << ": " << "No person detected yet" << std::endl;
            }
        }

        // 2. perform object tracking

        if (person_detected_first_time){
            cv::Rect rec{};
            bool tracker_update_ok = tracker->update(frame, rec);
            if (tracker_update_ok){
                // draw rectangle
                cv::rectangle(frame, rec, cv::Scalar(0, 255, 0), 2);
                log_file << "FRAME " << frame_count << ": " << "Target tracked: [" 
                    << "["<<rec.x<<","<<rec.y<<"],"
                    << "["<<rec.x+rec.width<<","<<rec.y<<"],"
                    << "["<<rec.x<<","<<rec.y+rec.height<<"],"
                    << "["<<rec.x+rec.width<<","<<rec.y+rec.height<<"]"
                    << "]"
                    << std::endl;
            } else {
                log_file << "FRAME " << frame_count << ": " << "Target being tracked lost" << std::endl;
            }
        }

        output_video_writer.write(frame);
        cv::imshow(WIN_NAME, frame);

        if (cv::waitKey(delay) == 27) { //ESC key
            break;
        }
    }

    log_file << "No more frames grabbed. Exiting..." << std::endl;
    log_file << "Total number of frames processed: " << frame_count << std::endl;
    int64_t end = cv::getTickCount();
    log_file << "Processing time: " << (end-start)/cv::getTickFrequency() << "s" << std::endl;
    cap.release();
    cv::destroyAllWindows();
    log_file.close();
}