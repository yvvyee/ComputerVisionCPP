#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char** argv) {
    Mat image, result;
    image = imread(argv[1], IMREAD_COLOR);

    cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
    kernel.at<float>(1, 1) = 5.0;
    kernel.at<float>(0, 1) = -1.0;
    kernel.at<float>(2, 1) = -1.0;
    kernel.at<float>(1, 0) = -1.0;
    kernel.at<float>(1, 2) = -1.0;

    filter2D(image, result, image.depth(), kernel);

    namedWindow("Original Image", WINDOW_AUTOSIZE);
    namedWindow("Filtered image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);
    imshow("Filtered image", result);

    waitKey(0);
    return 0;
}
