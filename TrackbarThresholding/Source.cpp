#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

/// 전역 변수 선언 ///
Mat image, src_gray, dst;

char window_name1[] = "AdaptiveThreshold Mean";
char window_name2[] = "AdaptiveThreshold Gauss";
char window_name3[] = "Threshold Demo";
char trackbar_type[] = "Type";
char trackbar_value[] = "Value";

int threshold_type = 0;
int threshold_value = 127;

int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

void Threshold_Demo(int, void*);

int main(int argc, char** argv) {

    image = imread(argv[1], IMREAD_COLOR);
    if (image.empty()) {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    cvtColor(image, src_gray, COLOR_BGR2GRAY);

    adaptiveThreshold(src_gray, dst, max_value, 
        ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 9);
    namedWindow(window_name1, WINDOW_AUTOSIZE);
    imshow(window_name1, dst);

    adaptiveThreshold(src_gray, dst, max_value, 
        ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 9);
    namedWindow(window_name2, WINDOW_AUTOSIZE);
    imshow(window_name2, dst);

    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", image);

    namedWindow(window_name3, WINDOW_AUTOSIZE);
    createTrackbar(trackbar_type, window_name3,
        &threshold_type, max_type, Threshold_Demo);
    createTrackbar(trackbar_value, window_name3,
        &threshold_value, max_value, Threshold_Demo);

    Threshold_Demo(0, 0);

    while (true) {
        int c;
        c = waitKey(20);
        if ((char)c == 27) {
            break;
        }
    }
}

void Threshold_Demo(int, void*) {
    /*
    0: Binary
    1: Binary Inverted
    2: Threshold Truncated
    3: Threshold to Zero
    4: Threshold to Zero Inverted
    */
    threshold(src_gray, dst, threshold_value,
        max_BINARY_value, threshold_type);
    imshow(window_name3, dst);
}
