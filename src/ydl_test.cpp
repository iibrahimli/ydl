#include <iostream>

#include "darknet.h"
#include "opencv2/opencv.hpp"

using namespace cv;



void parse_cfg(char *datacfg, char *cfgfile, char *weightfile){

    

}



int main() {

    VideoCapture cap(0);    // open the default camera
    if(!cap.isOpened())     // check if we succeeded
        return -1;

    std::cout << "Camera OK" << std::endl;

    namedWindow("capture");

    for(;;) {
        Mat frame;
        cap >> frame;

        if(frame.empty()) continue;

        imshow("capture", frame);

        if(waitKey(1) != 255) break;    // stop capturing by pressing ESC
    }

    cap.release();
    destroyAllWindows();

    return 0;
}