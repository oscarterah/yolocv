#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>

using namespace cv;
using namespace std;
using namespace dnn;

int main()
{
    cout<<CV_VERSION;
    VideoCapture cap;
    cap.open(0);

    std::string model = "yolov2.weights";
    std::string config = "yolov2.cfg";
    Net net = readNet(model, config,"Darknet");
    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(DNN_TARGET_OPENCL);

    vector<string> classNamesVec;
    ifstream classNamesFile("coco.names");
    if (classNamesFile.is_open())
    {
        string className = "";
        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }

    for (;;)
    {
        if (!cap.isOpened()) {
            cout << "Video Capture Fail" << endl;
            break;
        }
        Mat frame;
        cap >> frame;
        Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);
        net.setInput(inputBlob, "data");
        Mat detectionMat;
        net.forward(detectionMat);
        vector<double> layersTimings;
        double freq = getTickFrequency() / 1000;
        double time = net.getPerfProfile(layersTimings) / freq;
        ostringstream ss;
        ss << "FPS: " << 1000 / time << " ; time: " << time << " ms";
        putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));
        std::cout << time << std::endl;
        float confidenceThreshold = 0.1;    // rows represent number of detected object (proposed region)
        for (int i = 0; i < detectionMat.rows; i++)
        {
            const int probability_index = 5;
            const int probability_size = detectionMat.cols - probability_index;
            float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);

            size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

            if (confidence > confidenceThreshold)
            {
                float x = detectionMat.at<float>(i, 0);
                float y = detectionMat.at<float>(i, 1);
                float width = detectionMat.at<float>(i, 2);
                float height = detectionMat.at<float>(i, 3);
                int xLeftBottom = static_cast<int>((x - width / 2) * frame.cols);
                int yLeftBottom = static_cast<int>((y - height / 2) * frame.rows);
                int xRightTop = static_cast<int>((x + width / 2) * frame.cols);
                int yRightTop = static_cast<int>((y + height / 2) * frame.rows);

                Rect object(xLeftBottom, yLeftBottom,
                    xRightTop - xLeftBottom,
                    yRightTop - yLeftBottom);

                rectangle(frame, object, Scalar(0, 255, 0));

                if (objectClass < classNamesVec.size())
                {
                    ss.str("");
                    ss << confidence;
                    String conf(ss.str());
                    String label = String(classNamesVec[objectClass]) + ": " + conf;
                    int baseLine = 0;
                    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                    rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom),
                        Size(labelSize.width, labelSize.height + baseLine)),
                        Scalar(255, 255, 255), FILLED);
                    putText(frame, label, Point(xLeftBottom, yLeftBottom + labelSize.height),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
                }
                else
                {
                    cout << "Class: " << objectClass << endl;
                    cout << "Confidence: " << confidence << endl;
                    cout << " " << xLeftBottom
                        << " " << yLeftBottom
                        << " " << xRightTop
                        << " " << yRightTop << endl;
                }
            }
        }

        imshow("YOLO: Detections", frame);
        if (waitKey(1) >= 0) break;
    } 
        
    return 0;
}
