#include <iostream>
#include <opencv2/opencv.hpp>
#include "ydl/detector.hpp"

using std::cout;
using std::endl;
using namespace cv;


#define CAP_WIDTH  1280
#define CAP_HEIGHT 720



int main() {

    std::string cfg_filename     = "../model/yolov3_tiny_test.cfg";
    std::string weights_filename = "../model/yolov3_tiny_10k.weights";
    std::string names_filename   = "../model/objects.names";

    ydl::detector det(cfg_filename, weights_filename, names_filename);


    VideoCapture cap(0);     // open the default camera
    if(!cap.isOpened()){     // check if we succeeded
        cout << "Unable to open camera" << endl;
        return -1;
    }

    cout << "Camera OK" << std::endl;

    cap.set(CV_CAP_PROP_FRAME_WIDTH,  CAP_WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT);

    namedWindow("capture");

    // ydl::v_pred_result res;
    // ydl::duration dur;

    for(;;) {
        Mat frame;
        cap >> frame;

        if(frame.empty()) continue;

        // predict and annotate
        auto [res, dur] = det.predict(frame, 0.5);
        cout << res << endl;
        det.annotate(frame, res, dur);

        imshow("capture", frame);

        if(waitKey(1) != 255) break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}