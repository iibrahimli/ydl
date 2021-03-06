#include <iostream>
#include <opencv2/opencv.hpp>

#include "ydl/detector.hpp"

using std::cout;
using std::endl;
using namespace cv;


#define CAP_WIDTH  1280
#define CAP_HEIGHT 720

/*
    ############################################
    ############################################
    ####                                    ####
    ####  TODO: implement cv::MultiTracker  ####
    ####                                    ####
    ############################################
    ############################################
*/


int main() {

    std::string cfg_filename     = "../model/yolov3_tiny_test.cfg";
    std::string weights_filename = "../model/yolov3_tiny_10k.weights";
    std::string names_filename   = "../model/objects.names";

    ydl::detector det(cfg_filename, weights_filename, names_filename, ydl::HYBRID, 10);

    // open the default camera
    // VideoCapture cap(0);
    VideoCapture cap("../test/drones.mp4");

    if(!cap.isOpened()){
        cout << "Unable to open camera" << endl;
        return -1;
    }

    cout << "Camera OK" << std::endl;

    cap.set(CAP_PROP_FRAME_WIDTH,  CAP_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT);

    namedWindow("capture");

    for(;;){

        Mat frame;
        cap >> frame;

        if(frame.empty())
            continue;

        // predict and annotate
        auto [res, dur] = det.predict(frame, 0.5);
        if(res.size())
            cout << res << endl;
        det.annotate(frame, res, dur);

        imshow("capture", frame);

        if(waitKey(1) != -1)
            break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}