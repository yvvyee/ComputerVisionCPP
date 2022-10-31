#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	if (argc != 2) { return -1; }
	Mat src = imread(argv[1], IMREAD_COLOR);
	if (!src.data) { return -1; }

	namedWindow("src", WINDOW_FULLSCREEN);
	imshow("src", src);

	Mat rgbChannel[3];
	split(src, rgbChannel);

	namedWindow("Blue", WINDOW_FULLSCREEN);
	namedWindow("Green", WINDOW_FULLSCREEN);
	namedWindow("Red", WINDOW_FULLSCREEN);
	imshow("Blue", rgbChannel[0]);
	imshow("Green", rgbChannel[1]);
	imshow("Red", rgbChannel[2]);

	Mat zero_mat, fin_img;
	zero_mat = Mat::zeros(
		Size(src.cols, src.rows),
		CV_8UC1);
	{
		vector<Mat> channels;
		channels.push_back(zero_mat);
		channels.push_back(zero_mat);
		channels.push_back(rgbChannel[2]);
		merge(channels, fin_img);
		namedWindow("R", WINDOW_FULLSCREEN);
		imshow("R", fin_img);
	}
	{
		vector<Mat> channels;
		channels.push_back(zero_mat);
		channels.push_back(rgbChannel[1]);
		channels.push_back(zero_mat);
		merge(channels, fin_img);
		namedWindow("G", WINDOW_FULLSCREEN);
		imshow("G", fin_img);
	}
	{
		vector<Mat> channels;
		channels.push_back(rgbChannel[0]);
		channels.push_back(zero_mat);
		channels.push_back(zero_mat);
		merge(channels, fin_img);
		namedWindow("B", WINDOW_FULLSCREEN);
		imshow("B", fin_img);
	}
	waitKey(0);
	return 0;
}