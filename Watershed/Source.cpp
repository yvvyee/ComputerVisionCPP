#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], IMREAD_COLOR);
    if (src.empty()) { return -1; }
    imshow("Source Image", src);

    // 이후 거리 변환 과정에서 추출을 용이하게 하기 위해 배경을 어둡게 변환
    Mat mask;
    inRange(src, Scalar(255, 255, 255), 
        Scalar(255, 255, 255), mask);
    src.setTo(Scalar(0, 0, 0), mask);
    imshow("Black Background Image", src);

    // 샤프닝 커널
    Mat kernel = (Mat_<float>(3, 3) <<
        1, 1, 1,
        1, -8, 1,
        1, 1, 1);
    Mat imgLaplacian;
    filter2D(src, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

    imshow("Laplace Filtered Image", imgLaplacian);
    imshow("New Sharped Image", imgResult);

    // 이진화 이미지 생성
    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("Binary Image", bw);

    // 거리 변환 알고리즘
    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);

    // 다음 범위로 거리 변환 이미지를 정규화 = {0.0, 1.0}
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    imshow("Distance Transform Image", dist);

    // 전경 (foreground) 의 객체들을 구분하는 피크 지점을 얻기 위한 이진화
    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);

    // 피크 이미지 팽창
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);
    imshow("Dilated Peaks", dist);

    // findContours() 함수를 위한 8비트 범위로 변환
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    // 전체 마커의 외곽선 탐색
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 전경 마커 표시
    Mat markers = Mat::zeros(dist.size(), CV_32S);
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(markers, contours, static_cast<int>(i), 
            Scalar(static_cast<int>(i) + 1), -1);
    }

    // 배경 마커 표시
    circle(markers, Point(5, 5), 3, Scalar(255), -1);
    Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10);
    imshow("Markers_v1", markers8u);

    // watershed 알고리즘 수행
    watershed(imgResult, markers);
    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);

    imshow("Markers_v2", mark);
    
    // 무작위 색상값 생성
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    // 색상을 각 영역에 할당
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size())) {
                dst.at<Vec3b>(i, j) = colors[index - 1];
            }
        }
    }

    imshow("Separated Region Image", dst);
    waitKey();
    return 0;
}
