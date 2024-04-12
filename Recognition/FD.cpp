#include <opencv4/opencv2/opencv.hpp>

int main() {
    // Load the pre-trained model for face detection
    cv::dnn::Net net = cv::dnn::readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel");

    if (net.empty()) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }

    // Open the webcam
    // cv::VideoCapture cap(2);
    cv::VideoCapture cap;
    cap.open(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open webcam!" << std::endl;
        return 1;
    }

    // cv::Mat frame;
    while (true) {
        // Capture a frame from the webcam
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Failed to capture frame from webcam!" << std::endl;
            break;
        }

        // Convert the frame to a blob
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104, 177, 123));

        // Set the blob as input to the network
        net.setInput(blob);

        // Forward pass through the network
        cv::Mat detection = net.forward();
        std::cout<<"Detections: "<<detection<<std::endl;
        // Loop over the detected faces
        for (int i = 0; i < detection.size[2]; ++i) {
            float confidence = detection.at<float>(i, 2);

            if (confidence > 0.5) { // Threshold for face detection
                int x1 = static_cast<int>(detection.at<float>(i, 3) * frame.cols);
                int y1 = static_cast<int>(detection.at<float>(i, 4) * frame.rows);
                int x2 = static_cast<int>(detection.at<float>(i, 5) * frame.cols);
                int y2 = static_cast<int>(detection.at<float>(i, 6) * frame.rows);

                // Draw the bounding box around the detected face
                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            }
        }

        // Display the result
        cv::imshow("Face Detection", frame);

        // Exit the loop if the 'q' key is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the webcam and close the windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
