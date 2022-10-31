#include <opencv2/opencv.hpp>
using namespace cv;

// 영상 이진화 실습
int main(int argc, char** argv)
{
    // 사용자로부터 이미지 파일 경로를 입력받고 객체 생성 
    Mat image = imread(argv[1]);
    Mat result = Mat::zeros(image.size(), image.type());
    uchar Threshold = 0;

    std::cout << " threshold 입력 ( 0 ~ 255 ) : ";
    std::cin >> Threshold;

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (image.at<Vec3b>(i, j)[0] > Threshold) {
                result.at<Vec3b>(i, j)[0] = 255;
            }
            else {
                result.at<Vec3b>(i, j)[0] = 0;
            }
            if (image.at<Vec3b>(i, j)[1] > Threshold) {
                result.at<Vec3b>(i, j)[1] = 255;
            }
            else {
                result.at<Vec3b>(i, j)[1] = 0;
            }
            if (image.at<Vec3b>(i, j)[2] > Threshold) {
                result.at<Vec3b>(i, j)[2] = 255;
            }
            else {
                result.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }
    // 윈도우 생성 
    namedWindow("원본 이미지", 1);
    namedWindow("변환 이미지", 1);

    // 출력
    imshow("원본 이미지", image);
    imshow("변환 이미지", result);

    // 대기
    waitKey();
    return 0;
}