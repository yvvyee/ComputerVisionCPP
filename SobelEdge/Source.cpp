#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char** argv) {
    Mat image, result;
    image = imread(argv[1], IMREAD_GRAYSCALE);
    if (!image.data) return 0;

    cv::namedWindow("Original Image");
    cv::imshow("Original Image", image);

    cv::Mat sobelX;
    cv::Sobel(image, sobelX, CV_8U, 1, 0, 3, 0.4, 128);

    cv::namedWindow("Sobel X Image");
    cv::imshow("Sobel X Image", sobelX);

    cv::Mat sobelY;
    cv::Sobel(image, sobelY, CV_8U, 0, 1, 3, 0.4, 128);

    cv::namedWindow("Sobel Y Image");
    cv::imshow("Sobel Y Image", sobelY);

    cv::Sobel(image, sobelX, CV_16S, 1, 0);
    cv::Sobel(image, sobelY, CV_16S, 0, 1);

    cv::Mat sobel;
    sobel = abs(sobelX) + abs(sobelY);

    double sobmin, sobmax;
    cv::minMaxLoc(sobel, &sobmin, &sobmax);

    cv::Mat sobelImage;
    sobel.convertTo(sobelImage, CV_8U, -255. / sobmax, 255);

    cv::namedWindow("Sobel Image");
    cv::imshow("Sobel Image", sobelImage);

    cv::Mat sobelThresholded;
    cv::threshold(sobelImage, sobelThresholded, 225, 255, cv::THRESH_BINARY);

    cv::namedWindow("Binary Sobel Image (low)");
    cv::imshow("Binary Sobel Image (low)", sobelThresholded);

    waitKey(0);
    return 0;
}
