# face_detection_cpp

# How to compile and run an output

compile : g++ -o facedetection face_detection.cpp `pkg-config --cflags --libs opencv4`
run output : ./facedetection
