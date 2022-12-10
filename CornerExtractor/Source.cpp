#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv)
{
    cv::Mat image = cv::imread(argv[1], IMREAD_GRAYSCALE);
    cv::namedWindow("Original Image");
    cv::imshow("Original Image", image);

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(image, corners,
        500,    // 최대 개수
        0.1,    // 품질 레벨
        10);    // 코너점 간의 최소거리

    std::vector<cv::Point2f>::const_iterator it = corners.begin();
    while (it != corners.end()) {
        // 원형으로 표시
        cv::circle(image, *it, 3, cv::Scalar(255, 255, 255), 1);
        ++it;
    }

    cv::namedWindow("Good Features to Track");
    cv::imshow("Good Features to Track", image);
    cv::waitKey(0);

    return 0;
}
