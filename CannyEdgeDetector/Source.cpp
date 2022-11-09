#include <opencv2/opencv.hpp>
using namespace cv;

Mat src, src_gray;
Mat dst, src_edge;
char winname[] = "Edge Map";
int const minTH_Limit = 100;
int minTH = 0;
int ratio = 3;
int ksize = 3;


void CannyThreshold(int, void*) {
	Canny(src_gray, src_edge,
		minTH, minTH * ratio, ksize);

	dst = Scalar::all(0);

	src.copyTo(dst, src_edge);
	imshow(winname, dst);
}

int main(int argc, char** argv) {
	src = imread(argv[1], IMREAD_COLOR);
	if (!src.data) { return -1; }

	namedWindow("Original image");
	imshow("Original image", src);

	dst.create(src.size(), src.type());

	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	namedWindow(winname, WINDOW_AUTOSIZE);

	createTrackbar("Min", winname, 
		&minTH, minTH_Limit, CannyThreshold);

	CannyThreshold(0, 0);

	waitKey(0);
	return 0;
}
