#include <opencv2/opencv.hpp>

int main() {
    // Load pre-trained classifier for face detection (Haar Cascade classifier)
    cv::CascadeClassifier faceCascade;
    faceCascade.load(cv::samples::findFile("haarcascade_frontalface_default.xml"));

    // Open the default webcam (usually camera index 0)
    cv::VideoCapture video(0); // Change the parameter to specify a different camera if needed

    if (!video.isOpened()) {
        std::cerr << "Error: Couldn't access the webcam" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (video.read(frame)) {
        // Convert the frame to grayscale for face detection
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(grayFrame, grayFrame);

        // Detect faces in the frame
        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 3, 0, cv::Size(30, 30));

        // Draw rectangles around detected faces
        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
        }

        // Display the frame with detected faces
        cv::imshow("Face Detection", frame);

        // Exit the loop if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release video capture and close windows
    video.release();
    cv::destroyAllWindows();

    return 0;
}


// #include <opencv2/opencv.hpp>
// #include <iostream>

// int main() {
//     // Load pre-trained classifier for face detection (Haar Cascade classifier)
//     cv::CascadeClassifier faceCascade;
//     faceCascade.load(cv::samples::findFile("haarcascade_frontalface_default.xml"));

//     // Path to the input image
//     std::string imagePath = "download.jpeg"; // Replace with the path to your input image

//     // Read the input image
//     cv::Mat frame = cv::imread(imagePath);
//     if (frame.empty()) {
//         std::cerr << "Error: Could not read the input image" << std::endl;
//         return -1;
//     }

//     // Convert the image to grayscale for face detection
//     cv::Mat grayFrame;
//     cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
//     cv::equalizeHist(grayFrame, grayFrame);

//     // Detect faces in the image
//     std::vector<cv::Rect> faces;
//     faceCascade.detectMultiScale(grayFrame, faces, 1.1, 3, 0, cv::Size(30, 30));

//     // Draw rectangles around detected faces
//     for (const auto& face : faces) {
//         cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
//     }

//     // Display the image with detected faces
//     cv::imshow("Face Detection", frame);
//     cv::waitKey(0);

//     // Close the window after displaying the image
//     cv::destroyAllWindows();

//     return 0;
// }
