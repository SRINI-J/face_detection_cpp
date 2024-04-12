#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

#include <cmath>

double calculateCosineSimilarity(const std::vector<float>& embedding1, const std::vector<float>& embedding2) {
    if (embedding1.size() != embedding2.size() || embedding1.empty()) {
        return 0.0;  // Invalid or empty embeddings
    }

    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (size_t i = 0; i < embedding1.size(); ++i) {
        dotProduct += embedding1[i] * embedding2[i];
        normA += embedding1[i] * embedding1[i];
        normB += embedding2[i] * embedding2[i];
    }

    if (normA <= 0.0 || normB <= 0.0) {
        return 0.0;  // Avoid division by zero
    }

    return dotProduct / (sqrt(normA) * sqrt(normB));
}


static
void visualize(Mat& input, int frame, Mat& faces, double fps, std::vector<std::string>closestNames, int thickness = 2)
{
    // std::cout<<"Faces result: "<<faces<<std::endl;
    // int stopping;
    // std::cin>>stopping;
    std::string fpsString = cv::format("FPS : %.2f", (float)fps);
    if (frame >= 0)
        cout << "Frame " << frame << ", ";
    // cout << "FPS: " << fpsString << endl;
    for (int i = 0; i < faces.rows; i++)
    {
        // Print results
        cout << "Face " << i
             << ", top-left coordinates: (" << faces.at<float>(i, 0) << ", " << faces.at<float>(i, 1) << "), "
             << "box width: " << faces.at<float>(i, 2)  << ", box height: " << faces.at<float>(i, 3) << ", "
             << "score: " << cv::format("%.2f", faces.at<float>(i, 14))
             << endl;

        // Draw bounding box
        rectangle(input, Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), Scalar(0, 255, 0), thickness);
        // Draw landmarks
        circle(input, Point2i(int(faces.at<float>(i, 4)), int(faces.at<float>(i, 5))), 2, Scalar(255, 0, 0), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 6)), int(faces.at<float>(i, 7))), 2, Scalar(0, 0, 255), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 8)), int(faces.at<float>(i, 9))), 2, Scalar(0, 255, 0), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2, Scalar(255, 0, 255), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2, Scalar(0, 255, 255), thickness);
        putText(input, closestNames[i], Point(100, 155), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
    }
    putText(input, fpsString, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
}

// Function to print the embedding values
void printEmbedding(const Mat& embedding) {
    cout << "Embedding: ";
    for (int i = 0; i < embedding.cols; ++i) {
        cout << embedding.at<float>(0, i) << " ";
    }
    cout << endl;
}
// std::map<std::string, std::vector<std::vector<float>>>
void getNamesEmb() {
    // Initialize a map to store embeddings
    std::map<std::string, std::vector<double>> embeddingsMapStored;

    // Open the embeddings.txt file
    std::ifstream infile("embeddings.txt");
    if (!infile.is_open()) {
        std::cerr << "Failed to open embeddings.txt" << std::endl;
        // return 1;
    }

    // Read lines from the file
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string name;

        // Extract name
        if (!(iss >> name)) {
            continue;  // Skip empty lines
        }

        // Get the rest of the line as embeddings
        std::string embeddingString;
        std::getline(iss >> std::ws, embeddingString);
        std::istringstream embeddingStream(embeddingString);

        double val;
        std::vector<double> embeddings;
        while (embeddingStream >> val) {
            embeddings.push_back(val);
        }

        embeddingsMapStored[name] = embeddings;
    }

    // Close the file
    infile.close();
    std::cout<<"Length: "<<embeddingsMapStored.size()<<std::endl;
    // Print the stored embeddings
    for (const auto& pair : embeddingsMapStored) {
        std::cout << pair.first << ": ";
        for (double val : pair.second) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv,
        "{help  h           |            | Print this message}"
        "{image1 i1         |            | Path to the input image1. Omit for detecting through VideoCapture}"
        "{image2 i2         |            | Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm}"
        // "{video v           | 0          | Path to the input video}"
        "{video v           | v4l2src device=/dev/video0 ! videoconvert ! appsink          | Path to the input video}"
        "{scale sc          | 1.0        | Scale factor used to resize input video frames}"
        "{fd_model fd       | face_detection_yunet_2023mar.onnx| Path to the model. Download yunet.onnx in https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet}"
        "{fr_model fr       | face_recognition_sface_2021dec.onnx | Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface}"
        "{score_threshold   | 0.9        | Filter out faces of score < score_threshold}"
        "{nms_threshold     | 0.3        | Suppress bounding boxes of iou >= nms_threshold}"
        "{top_k             | 5000       | Keep top_k bounding boxes before NMS}"
        "{save s            | false      | Set true to save results. This flag is invalid when using camera}"
    );
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    // Map to store embeddings for different persons
    std::map<std::string, std::vector<std::vector<float>>> embeddingsMap;
    int similarityThreshold = 0.5;
    String fd_modelPath = parser.get<String>("fd_model");
    String fr_modelPath = parser.get<String>("fr_model");

    float scoreThreshold = parser.get<float>("score_threshold");
    float nmsThreshold = parser.get<float>("nms_threshold");
    int topK = parser.get<int>("top_k");

    bool save = parser.get<bool>("save");
    float scale = parser.get<float>("scale");

    double cosine_similar_thresh = 0.363;
    double l2norm_similar_thresh = 1.128;

    //! [initialize_FaceDetectorYN]
    // Initialize FaceDetectorYN
    Ptr<FaceDetectorYN> detector = FaceDetectorYN::create(fd_modelPath, "", Size(320, 320), scoreThreshold, nmsThreshold, topK);
    //! [initialize_FaceDetectorYN]

    TickMeter tm;

    // Initialize FaceRecognizerSF
    Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create(fr_modelPath, "");


    // If input is an image
    if (parser.has("image1"))
    {
        cout<<"For Face Recognition."<<std::endl;
    }
    else
    {
        bool saveEmbedding = false;
        std::string currentPerson;

        int frameWidth, frameHeight;
        VideoCapture capture;
        std::string video = parser.get<string>("video");
        if (video.size() == 1 && isdigit(video[0]))
            capture.open(parser.get<int>("video"));
        else
            capture.open(samples::findFileOrKeep(video));  // keep GStreamer pipelines
        if (capture.isOpened())
        {
            frameWidth = int(capture.get(CAP_PROP_FRAME_WIDTH) * scale);
            frameHeight = int(capture.get(CAP_PROP_FRAME_HEIGHT) * scale);
            cout << "Video " << video
                << ": width=" << frameWidth
                << ", height=" << frameHeight
                << endl;
        }
        else
        {
            cout << "Could not initialize video capturing: " << video << "\n";
            return 1;
        }

        detector->setInputSize(Size(frameWidth, frameHeight));

        cout << "Press 'SPACE' to save frame, any other key to exit..." << endl;
        int nFrame = 0;
        std::vector<std::string> closestNames;
        for (;;)
        {
            // Get frame
            Mat frame;
            if (!capture.read(frame))
            {
                cerr << "Can't grab frame! Stop\n";
                break;
            }

            resize(frame, frame, Size(frameWidth, frameHeight));

            // Inference
            Mat faces;
            tm.start();
            detector->detect(frame, faces);
            tm.stop();
            
            int key = waitKey(1);
            bool saveFrame = save;
            // Iterate over detected faces and print embeddings
            
            for (int i = 0; i < faces.rows; ++i) {
                Mat aligned_face;
                faceRecognizer->alignCrop(frame, faces.row(i), aligned_face);
                Mat feature;
                faceRecognizer->feature(aligned_face, feature);
                // printEmbedding(feature);

                getNamesEmb();

                // Load stored embeddings from "embeddings.txt" file
                std::ifstream embeddingsFileStored("embeddings.txt");
                std::map<std::string, std::vector<std::vector<float>>> embeddingsMapStored;
                // std::cout<<"Opened or Not------------------"<<embeddingsFileStored.is_open()<<std::endl;
                if (embeddingsFileStored.is_open()) {
                    std::string line;
                    while (std::getline(embeddingsFileStored, line)) {
                        std::istringstream iss(line);
                        std::string name;
                        std::vector<float> embedding;

                        iss >> name >> std::ws;  // Read the name
                        float value;
                        while (iss >> value) {
                            embedding.push_back(value);
                        }

                        embeddingsMapStored[name].push_back(embedding);
                    }
                    embeddingsFileStored.close();
                    cout << "Loaded stored embeddings from 'embeddings.txt'." << std::endl;
                } else {
                    cerr << "Unable to open 'embeddings.txt' for reading." << std::endl;
                }

                std::cout<<"Printing Stored names."<<embeddingsMapStored.size()<<std::endl;
                for(auto item : embeddingsMapStored){
                    std::cout<<"Name: "<<item.first<<std::endl;
                }
                // int stopping;
                // std::cin>>stopping;
                // Compare the current embedding with stored embeddings
                std::string closestName;
                double maxSimilarity = 0.0;

                for (const auto& entry : embeddingsMap) {
                    for (const auto& storedEmbedding : entry.second) {
                        double similarity = calculateCosineSimilarity(feature, storedEmbedding);  // Implement similarity calculation here
                        if (similarity > maxSimilarity) {
                            maxSimilarity = similarity;
                            closestName = entry.first;
                        }
                    }
                }

                if (maxSimilarity > similarityThreshold) {
                    cout << "Detected face belongs to: " << closestName << " with similarity: " << maxSimilarity << endl;
                    // Now you can associate the detected face with the closestName
                    // and perform other relevant actions.
                    closestNames.push_back(closestName);
                } else {
                    closestNames.push_back("Unknown");
                }
        // }   
                
                if (key == ' ')
                {
                    saveFrame = true;
                    key = 0;  // handled
                } else if (key == 's') {
                    // saveEmbedding = true;
                    cout << "Enter person's name: ";
                    std::cin >> currentPerson;
                    key = 0;

                    if (!currentPerson.empty()) {
                        embeddingsMap[currentPerson].emplace_back(
                            std::vector<float>(feature.begin<float>(), feature.end<float>())
                        );
                    }
                    // Save embeddings to a file
                    std::ofstream embeddingsFile("embeddings.txt", std::ios::app);
                    // std::cout<<"Saved----------------"<<embeddingsFile.is_open()<<std::endl;
                    if (embeddingsFile.is_open()) {
                        // std::cout<<"First---------"<<std::endl;
                        for (const auto& entry : embeddingsMap) {
                            embeddingsFile << entry.first << ": ";
                            for (const auto& embedding : entry.second) {
                                for (float value : embedding) {
                                    embeddingsFile << value << " ";
                                }
                                embeddingsFile << "| ";
                            }
                            embeddingsFile << "\n";
                        }
                        embeddingsFile.close();
                        cout << "Embeddings saved to 'embeddings.txt'." << std::endl;
                    } else {
                        cerr << "Unable to open 'embeddings.txt' for writing." << std::endl;
                    }
                    // saveEmbedding = false;
                }
            }

            
            Mat result = frame.clone();
            // Draw results on the input image
            visualize(result, nFrame, faces, tm.getFPS(), closestNames);

            // Visualize results
            imshow("Live", result);
            for(int i =0;i<closestNames.size();i++){
                std::cout<<"Inside closestnames-------------------"<<std::endl;
                std::cout<<closestNames[i]<<std::endl;
            }
            

            if (saveFrame)
            {
                std::string frame_name = cv::format("frame_%05d.png", nFrame);
                std::string result_name = cv::format("result_%05d.jpg", nFrame);
                cout << "Saving '" << frame_name << "' and '" << result_name << "' ...\n";
                imwrite(frame_name, frame);
                imwrite(result_name, result);
                saveFrame = false;
            }

            ++nFrame;

            if (key > 0)
                break;
        }
        cout << "Processed " << nFrame << " frames" << endl;
    }
    cout << "Done." << endl;
    return 0;
}
