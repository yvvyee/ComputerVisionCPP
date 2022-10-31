#include <opencv2/opencv.hpp>
using namespace cv;

void SaltPepper(Mat& img, int n)
{
    for (int k = 0; k < n; k++) {
        int i = rand() % img.cols;
        int j = rand() % img.rows;

        if (img.channels() == 1)
        {	// Gray scale image
            img.at<uchar>(j, i) = 255;
        }
        else if (img.channels() == 3)
        {	// Color image
            img.at<Vec3b>(j, i)[0] = 255;
            img.at<Vec3b>(j, i)[1] = 255;
            img.at<Vec3b>(j, i)[2] = 255;
        }
    }
}

int main(int argc, char** argv) {
    // 주석
    Mat image, result, dst, dst1;
    image = imread(argv[1], IMREAD_COLOR);

    SaltPepper(image, 30000);

    cv::namedWindow("S&P Image");
    cv::imshow("S&P Image", image);

    cv::blur(image, result, cv::Size(5, 5));

    cv::namedWindow("Mean filtered S&P Image");
    cv::imshow("Mean filtered S&P Image", result);

    cv::medianBlur(image, result, 5);

    cv::namedWindow("Median filtered S&P Image");
    cv::imshow("Median filtered S&P Image", result);

    waitKey(0);
    return 0;
}