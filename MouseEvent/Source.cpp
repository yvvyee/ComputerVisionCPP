#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;
void MyCallback(int event,
	int x, int y, int flags, void* param);
int main(int argc, char** argv) {
	int i, j, k;
	Mat image = imread(argv[1], IMREAD_COLOR);
	if (!image.data) { return -1; }
	namedWindow("Main", WINDOW_AUTOSIZE);
	setMouseCallback("Main", MyCallback, &image);
	imshow("Main", image);
	waitKey(0);
	return 0;
}
void MyCallback(int event,
	int x, int y, int flags, void* param)
{
	Mat* image = (Mat *)param;
	int thickness = -1;
	int lineType = 8;

	if (event == EVENT_MOUSEMOVE)
	{
		cout << "Moved (" << x 
			<< ", " << y << ")" << endl;
	} else
	if (event == EVENT_RBUTTONDOWN) {
		cout << "RButton (" << x
			<< ", " << y << ")" << endl;
	} else
	if (event == EVENT_LBUTTONDOWN) {
		printf("LButton (%d, %d) [B: %d, G: %d, R: %d]\n",
			x, y,
			(int)(*image).at<Vec3b>(y, x)[0],
			(int)(*image).at<Vec3b>(y, x)[1],
			(int)(*image).at<Vec3b>(y, x)[2]);
	}
}