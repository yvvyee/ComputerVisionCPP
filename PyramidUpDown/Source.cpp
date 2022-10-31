#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char** argv) {
    Mat src, dst, tmp;
    const char* window_name = "Pyramids Demo";

    printf("\n Zoom In-Out demo  \n ");
    printf("------------------ \n");
    printf(" * [u] -> Zoom in  \n");
    printf(" * [d] -> Zoom out \n");
    printf(" * [ESC] -> Close program \n \n");

    /// Test image - Make sure it s divisible by 2^{n}
    src = imread(argv[1], IMREAD_COLOR);
    if (!src.data) {
        printf(" No data! -- Exiting the program \n");
        return -1;
    }

    tmp = src;
    dst = tmp;

    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", src);

    while (true) {
        int c;
        c = waitKey(10);

        if ((char)c == 27) { break; }
        if ((char)c == 'u')
        {
            pyrUp(tmp, dst, Size(tmp.cols * 2, tmp.rows * 2));
            printf("** Zoom In: Image x 2 \n");
        }
        else if ((char)c == 'd') {
            pyrDown(tmp, dst, Size(tmp.cols / 2, tmp.rows / 2));
            printf("** Zoom Out: Image / 2 \n");
        }

        imshow(window_name, dst);
        tmp = dst;
    }
    return 0;
}