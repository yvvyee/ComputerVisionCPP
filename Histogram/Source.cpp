#include <iostream>
#include "Histogram1D.h"
using namespace std;

int main(int argc, char** argv) {
    Mat image_grey, image_color, image_equal;

    /// Load image
    image_grey = imread(argv[1], IMREAD_GRAYSCALE);
    image_color = imread(argv[1], IMREAD_COLOR);

    if (image_grey.empty()) {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    Histogram1D h; // 히스토그램을 위한 객체
    MatND histo = h.getHistogram(image_grey); // 히스토그램 계산
    MatND Chist = h.getHistogramCImage(image_color);
    image_equal = h.equalize(image_grey);
    MatND equal = h.getHistogramImage(image_equal);

    for (int i = 0; i < 256; i++) // 히스토그램의 빈도를 조회
        std::cout << "Value" << i << "=" <<
        histo.at<float>(i) << std::endl;

    //영상을 두 그룹으로 나누는 부분을 경계값으로 처리해 확인
    Mat thresholdedImage; // 경계값으로 이진 영상 생성
    threshold(image_grey, thresholdedImage, 60, 
        255, THRESH_BINARY);

    namedWindow("Greyscale", WINDOW_AUTOSIZE);
    namedWindow("Color", WINDOW_AUTOSIZE);
    namedWindow("Equalize", WINDOW_AUTOSIZE);
    namedWindow("EqualizeHistogram");
    namedWindow("Greyscale Histogram");
    namedWindow("Binary Image");
    namedWindow("Color Histogram");

    imshow("Greyscale", image_grey);
    imshow("Color", image_color);
    imshow("Greyscale Histogram", h.getHistogramImage(image_grey));
    imshow("Binary Image", thresholdedImage);
    imshow("Color Histogram", Chist);
    imshow("Equalize", image_equal);
    imshow("EqualizeHistogram", equal);

    waitKey(0);
    return 0;

}

