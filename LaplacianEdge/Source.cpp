#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char** argv) {
    Mat src, src_gray, dst;
    src = imread(argv[1], IMREAD_COLOR);
    if (!src.data) { return -1; }

    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    const char* window_name = "Laplace Demo";
    const char* window_name1 = "Original Image";
    int c;

    namedWindow(window_name1, WINDOW_AUTOSIZE);
    imshow(window_name1, src);
    GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    Mat abs_dst;
    Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(dst, abs_dst);

    namedWindow(window_name, WINDOW_AUTOSIZE);
    imshow(window_name, abs_dst);
    waitKey(0);
    return 0;
}