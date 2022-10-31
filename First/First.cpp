#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	Mat image;

	char* imageName = argv[1];

	image = imread(imageName, IMREAD_COLOR);
	if (argc != 2 || !image.data) {
		printf(" No image data \n ");
		return -1;
	}
	Mat gray_image;
	cvtColor(image, gray_image, COLOR_BGR2GRAY);

	namedWindow(imageName, WINDOW_AUTOSIZE);
	namedWindow("Gray image", WINDOW_AUTOSIZE);

	imshow(imageName, image);
	imshow("Gray image", gray_image);

	waitKey(0);


	return 0;
}
