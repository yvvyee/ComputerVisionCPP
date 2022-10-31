#pragma warning(disable:4996)
#include <opencv2/opencv.hpp>

using namespace cv;

const int alpha_slider_max = 100;
int alpha_slider;
double alpha;
double beta;

Mat image1, image2, dst;

void on_trackbar(int, void*);

int main(int argc, char** argv)
{
    image1 = imread(argv[1], IMREAD_COLOR);
    image2 = imread(argv[2], IMREAD_COLOR);

    namedWindow("Display Blend");

    char TrackbarName[50];
    sprintf(TrackbarName, "Alpha x %d", alpha_slider_max);

    createTrackbar(TrackbarName, 
        "Display Blend", 
        &alpha_slider,
        alpha_slider_max, 
        on_trackbar);

    on_trackbar(alpha_slider, 0);

    waitKey(0);
    return 0;
}

void on_trackbar(int, void*)
{
    alpha = (double) alpha_slider/alpha_slider_max ;
    beta = ( 1.0 - alpha );
    addWeighted( image1, alpha, image2, beta, 0.0, dst);
    imshow("Display Blend", dst);
}