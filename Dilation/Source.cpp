#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    string window_name = "Dilation Example";
    Mat image, image_gray, dst, dst1;
    int threshold_value = 127;
    int const threshold_type = 4;
    int const max_BINARY_value = 255;

    image = imread(argv[1], IMREAD_COLOR);
    if (image.empty()) {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    cvtColor(image, image_gray, COLOR_BGR2GRAY);

    namedWindow(window_name, WINDOW_AUTOSIZE);
    imshow(window_name, image_gray);

    threshold(image_gray, dst, threshold_value = 120,
        max_BINARY_value, threshold_type);

    namedWindow("Binary image", WINDOW_AUTOSIZE);
    imshow("Binary image", dst);

    dilate(dst, dst1, Mat());
    namedWindow("Dilated image", WINDOW_AUTOSIZE);
    imshow("Dilated image", dst1);

    dilate(dst, dst1, Mat(), Point(-1, -1), 5);
    namedWindow("Dilated image: 5 times", WINDOW_AUTOSIZE);
    imshow("Dilated image: 5 times", dst1);

    waitKey(0);
    return 0;
}

