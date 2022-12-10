#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    Mat src = imread(argv[1], IMREAD_COLOR);
    if (src.empty()) { return -1; }
    imshow("Input Source Image", src);

    // 이진 이미지로 변환
    Mat thresh;
    cvtColor(src, thresh, COLOR_BGR2GRAY);
    threshold(thresh, thresh, 0, 255,
        THRESH_BINARY_INV | THRESH_OTSU);
    imshow("Binary Image", thresh);

    // 노이즈 제거를 위한 열림 연산
    Mat opening;
    Mat kernel = Mat::ones(3, 3, CV_8U);
    morphologyEx(thresh, opening, MORPH_OPEN, kernel, Point(-1, -1), 2);
    imshow("De-noised Image", opening);

    // 전경 (객체) 과 배경을 구분
    Mat sure_bg;
    dilate(opening, sure_bg, kernel, Point(-1, -1), 3);
    imshow("Background Image", sure_bg);
    
    // skeleton 또는 thinning 이미지를 얻기 위한 거리 변환
    // 전경의 특정 객체의 중심에서 점점 옅어져 가는 형태
    // 전경 객체를 명확하게 구분할 수 있음
    Mat dist;
    distanceTransform(opening, dist, DIST_L2, 3);
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    imshow("Distance Transform Image", dist);
    
    // 이진화를 통해 개별적인 객체들만 남김
    Mat sure_fg;
    double max, min;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(dist, &min, &max, &min_loc, &max_loc);
    threshold(dist, sure_fg, 0.5 * max, 255, THRESH_BINARY);
    sure_fg.convertTo(sure_fg, CV_8U);

    imshow("Foreground Image", sure_fg);

    // 배경 이미지에서 전경 이미지의 영역을 제외
    Mat unknown;
    subtract(sure_bg, sure_fg, unknown);

    imshow("Unknown Image", unknown);

    // 전경 객체에 레이블링 작업
    Mat markers1;
    connectedComponents(sure_fg, markers1);
    markers1 += 1;
    markers1.setTo(0, unknown);

    // countour 계산을 위해 8비트로 변환
    Mat markers_8u;
    markers1.convertTo(markers_8u, CV_8U);
    
    imshow("Markers V1", markers_8u);

    // 전체 마커의 외곽선 탐색
    vector<vector<Point> > contours;
    findContours(markers_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // watershed 알고리즘 수행
    watershed(src, markers1);
    Mat markers2;
    markers1.convertTo(markers2, CV_8U);
    bitwise_not(markers2, markers2);

    imshow("Markers V2", markers2);

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
    Mat dst = Mat::zeros(markers1.size(), CV_8UC3);
    for (int i = 0; i < markers1.rows; i++) {
        for (int j = 0; j < markers1.cols; j++) {
            int index = markers1.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size())) {
                dst.at<Vec3b>(i, j) = colors[index - 1];
            }
        }
    }

    imshow("Separated Region Image", dst);
    waitKey();
    return 0;
}
