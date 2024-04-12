#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>

int main() {
    cv::VideoCapture capture("v4l2src device=/dev/video0 ! videoconvert ! appsink", cv::CAP_GSTREAMER);
    
    if (!capture.isOpened()) {
        std::cout << "Could not initialize video capturing!" << std::endl;
        return 1;
    }
    
    cv::Mat frame;
    while (true) {
        if (!capture.read(frame)) {
            std::cerr << "Can't grab frame! Stop" << std::endl;
            break;
        }
        
        cv::imshow("Live", frame);
        
        int key = cv::waitKey(1);
        if (key == 27)  // Press 'Esc' to exit
            break;
    }
    
    capture.release();
    cv::destroyAllWindows();
    
    return 0;
}
