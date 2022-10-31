#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char** argv) {
    Mat image;
    image = imread(argv[1], IMREAD_COLOR);

    cv::namedWindow("Orignal image");
    cv::imshow("Orignal image", image);

    // Unsharp Mask 를 이용한 샤프닝 기법
    Mat blurred; double sigma = 1, amount = 3;
    GaussianBlur(image, blurred, Size(), sigma, sigma);
    Mat sharpened = image * (1 + amount) + blurred * (-amount);

    cv::namedWindow("Sharpened image");
    cv::imshow("Sharpened image", sharpened);

    waitKey(0);
    return 0;
}