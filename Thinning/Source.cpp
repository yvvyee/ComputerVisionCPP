#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    string window_name = "Thinning Example";
    Mat image_gray, dst, dst1;
    int threshold_value = 127;
    int const threshold_type = 4;
    int const max_BINARY_value = 255;

    image_gray = imread(argv[1], IMREAD_GRAYSCALE);
    if (image_gray.empty()) {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    threshold(image_gray, image_gray, threshold_value = 120,
        max_BINARY_value, THRESH_BINARY);

    namedWindow("Binary image", WINDOW_AUTOSIZE);
    imshow("Binary image", image_gray);

    Mat skel(image_gray.size(), CV_8UC1, cv::Scalar(0));
    Mat temp(image_gray.size(), CV_8UC1);

    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_CROSS, cv::Size(3, 3));

    bool done;

    do {
        cv::morphologyEx(image_gray, temp,
            cv::MORPH_OPEN, element);
        cv::bitwise_not(temp, temp);
        cv::bitwise_and(image_gray, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        cv::erode(image_gray, image_gray, element);

        double max;
        cv::minMaxLoc(image_gray, 0, &max);
        done = (max == 0);
    } while (!done);
    imshow("Skeleton", skel);
    waitKey(0);
    return 0;
}