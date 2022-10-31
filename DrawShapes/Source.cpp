#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int window_size = 512;
void MyLine(Mat img, Point start, Point end);
void MyEllipse(Mat img, double angle);
void MyFilledCircle(Mat img, Point center);
int main(void)
{
	Mat image(512, 512,
		CV_8UC3, Scalar(0, 0, 0));
	Mat result;

	rectangle(image,
		Point(100, 100),
		Point(400, 400),
		CV_RGB(0, 255, 0));
	line(image, Point(400, 100),
		Point(100, 400), Scalar(0, 255, 0));

	namedWindow("Rectangle", WINDOW_AUTOSIZE);
	imshow("Rectangle", image);
	waitKey(0);
	//return 0;

	int k = 0;
	char atom_window[] = "Drawing : 1 : Atom";

	Mat atom_img = Mat::zeros(window_size,
		window_size, CV_8UC3);
	namedWindow(atom_window, WINDOW_AUTOSIZE);

	cout << "1 - 타원" << endl;
	cout << "2 - 채워진 원" << endl;
	cout << "3 - 사각형" << endl;
	cout << "4 - 라인" << endl;

	k = waitKey(0);
	if (k == 49)
	{
		MyEllipse(atom_img, 90);
		MyEllipse(atom_img, 0);
		MyEllipse(atom_img, 45);
		MyEllipse(atom_img, -45);

	} else
	if (k == 50)
	{
		MyFilledCircle(atom_img,
			Point(window_size / 2,
				window_size / 2));
	} else
	if (k == 51)
	{
		rectangle(atom_img,
			Point(0.7 * window_size / 8.0),
			Point(window_size, window_size),
			Scalar(0, 255, 0), -1, 8);
	} else
	if (k == 52)
	{
		MyLine(atom_img,
			Point(0, 15 * window_size / 16),
			Point(window_size,
				15 * window_size / 16));
	}
	imshow(atom_window, atom_img);
	waitKey();
	return 0;
}

void MyLine(Mat img, Point start, Point end)
{
	int thickness = 2;
	int lineType = 8;
	line(img, start, end,
		Scalar(0, 0, 255),
		thickness, lineType);
}

void MyEllipse(Mat img, double angle)
{
	int thickness = 2;
	int lineType = 8;

	ellipse(img,
		Point(window_size / 2., window_size / 2.),
		Size(window_size / 4., window_size / 16.),
		angle,
		0,
		360,
		Scalar(255, 0, 0),
		thickness,
		lineType
	);
}

void MyFilledCircle(Mat img,
	Point center)
{
	int thickness = -1;
	int lineType = 8;
	circle(img, center, window_size / 6.0,
		Scalar(0, 0, 255), thickness, lineType);
}