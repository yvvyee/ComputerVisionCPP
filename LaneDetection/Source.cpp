#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char** argv) {
	VideoCapture cap(argv[1]);
	
	if (!cap.isOpened()) {
		return -1;
	}
	Mat img;
	while (true)
	{
		cap >> img;
		if (img.empty())
		{
			printf("empty image");
			return 0;
		}
		imshow("camera img", img);
		if (waitKey(1) == 27) break;
	}
	return 0;
}