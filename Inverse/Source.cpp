#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) 
{
	Mat image, result;
	image = imread("C:/Users/ywlee/Pictures/test.jpg", IMREAD_COLOR);
	result = image.clone();

	if (image.empty())
	{
		cout << "못 열었습니다." << endl;
		return -1;
	}
	namedWindow("window", WINDOW_AUTOSIZE);
	imshow("window", image);

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			result.at<Vec3b>(i, j)[0] = 255 - image.at<Vec3b>(i, j)[0];
			result.at<Vec3b>(i, j)[1] = 255 - image.at<Vec3b>(i, j)[1];
			result.at<Vec3b>(i, j)[2] = 255 - image.at<Vec3b>(i, j)[2];
		}
	}

	namedWindow("window2", WINDOW_AUTOSIZE);
	imshow("window2", result);

	waitKey(0);
	destroyAllWindows();
	return 0;
}