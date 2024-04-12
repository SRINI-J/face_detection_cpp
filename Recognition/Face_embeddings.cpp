#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // Load the pre-trained model for face detection
    std::string detection_model_path = "res10_300x300_ssd_iter_140000.caffemodel";
    std::string detection_config_path = "deploy.prototxt";
    cv::dnn::Net detection_net = cv::dnn::readNetFromCaffe(detection_config_path, detection_model_path);

    // Load the pre-trained model for face recognition (FaceNet)
    std::string recognition_model_path = "nn4.small2.v1.t7";
    cv::dnn::Net recognition_net = cv::dnn::readNetFromTorch(recognition_model_path);

    // Check if the models loaded successfully
    if (detection_net.empty() || recognition_net.empty()) {
        std::cerr << "Error: Unable to load the models." << std::endl;
        return -1;
    }

    // Open the default webcam
    cv::VideoCapture cap(0);

    // Check if the webcam opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open the webcam." << std::endl;
        return -1;
    }

    // Create a vector to store the face embeddings and corresponding face rectangles
    std::vector<cv::Mat> face_embeddings;
    std::vector<cv::Rect> face_rectangles;
    std::vector<std::string> face_names;

    while (true) {
        // Capture a frame from the webcam
        cv::Mat frame;
        cap >> frame;

        // Create a blob from the frame and set it as the input to the face detection network
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104, 177, 123));
        detection_net.setInput(blob);

        // Forward pass through the face detection network to get the detections
        cv::Mat detections = detection_net.forward();

        // Process the output and obtain face embeddings
        for (int i = 0; i < detections.size[2]; ++i) {
            float confidence = detections.at<float>(i, 2);
            if (confidence > 0.5) {
                int y1 = static_cast<int>(detections.at<float>(i, 4) * frame.rows);
                int x1 = static_cast<int>(detections.at<float>(i, 3) * frame.cols);
                int x2 = static_cast<int>(detections.at<float>(i, 5) * frame.cols);
                int y2 = static_cast<int>(detections.at<float>(i, 6) * frame.rows);

                // Ensure the bounding box coordinates are within the image dimensions
                x1 = std::max(0, x1);
                y1 = std::max(0, y1);
                x2 = std::min(frame.cols - 1, x2);
                y2 = std::min(frame.rows - 1, y2);

                // Verify that the ROI coordinates and dimensions are valid
                if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 &&
                    x2 <= frame.cols && y2 <= frame.rows) {
                    // Extract the face ROI from the frame
                    cv::Mat face = frame(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));

                    // Preprocess the face for face recognition (resize to 96x96)
                    cv::Mat face_blob = cv::dnn::blobFromImage(face, 1.0 / 255, cv::Size(96, 96), cv::Scalar(0, 0, 0), false, false);

                    // Set the face blob as input to the face recognition network
                    recognition_net.setInput(face_blob);

                    // Forward pass through the face recognition network to get the face embeddings
                    cv::Mat embeddings = recognition_net.forward();

                    // The embeddings contain the face representation as a vector of floats (128-dimensional)
                    // You can store these embeddings for face recognition purposes
                    face_embeddings.push_back(embeddings.clone());
                    face_rectangles.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));

                    // Prompt the user to enter the name for the detected face
                    std::string name;
                    std::cout << "Enter the name for the detected face: ";
                    std::cin >> name;
                    face_names.push_back(name);

                    // Draw the bounding box and name around the face in the frame
                    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                    cv::putText(frame, name, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                }
            }
        }

        // Display the frame with detected faces
        cv::imshow("Detected Faces", frame);

        // Break the loop if the 'q' key is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the webcam and close the window
    cap.release();
    cv::destroyAllWindows();

    // Save the face embeddings and corresponding face rectangles to files
    // For example, you can save them as XML or YAML files
    cv::FileStorage fs("face_embeddings.xml", cv::FileStorage::WRITE);
    if (fs.isOpened()) {
        for (size_t i = 0; i < face_embeddings.size(); ++i) {
            cv::write(fs, "embedding" + std::to_string(i), face_embeddings[i]);
            cv::write(fs, "rectangle" + std::to_string(i), face_rectangles[i]);
            cv::write(fs, "name" + std::to_string(i), face_names[i]);
        }
        fs.release();
        std::cout << "Face embeddings and names saved to file." << std::endl;
    } else {
        std::cerr << "Error: Unable to open file for writing face embeddings and names." << std::endl;
    }

    return 0;
}
