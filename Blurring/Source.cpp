#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char** argv) {
    Mat result, image = imread(argv[1], IMREAD_COLOR);
    namedWindow("Original Image");
    imshow("Original Image", image);

    blur(image, result, cv::Size(5, 5));

    namedWindow("Mean filtered Image");
    imshow("Mean filtered Image", result);

    GaussianBlur(image, result, cv::Size(5, 5), 1.5);

    namedWindow("Gaussian filtered Image");
    imshow("Gaussian filtered Image", result);

    Mat gauss = cv::getGaussianKernel(9, 1.5, CV_32F);

    Mat_<float>::const_iterator it = gauss.begin<float>();
    Mat_<float>::const_iterator itend = gauss.end<float>();
    std::cout << "[";
    for (; it != itend; ++it) {
        std::cout << *it << " ";
    }
    std::cout << "]" << std::endl;

    waitKey();
    return 0;
}
