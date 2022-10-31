#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

Mat src, img, ROI;
Rect cropRect(0, 0, 0, 0);
Point P1(0, 0);
Point P2(0, 0);

const char* winName = "Cropped Image";
bool clicked = false;
int i = 0;
char imgName[15];

void checkBoundary()
{
	if (cropRect.width > img.cols - cropRect.x)
	{
		cropRect.width = img.cols - cropRect.x;
	}
	if (cropRect.height > img.rows - cropRect.y)
	{
		cropRect.height = img.rows - cropRect.y;
	}
	if (cropRect.x < 0)
	{
		cropRect.x = 0;
	}
	if (cropRect.y < 0)
	{
		cropRect.y = 0;
	}
}
void showImage()
{
	// 주석 테스트
	img = src.clone();
	checkBoundary();
	if (cropRect.width > 0 && cropRect.height > 0)
	{
		ROI = src(cropRect);
		imshow("cropped", ROI);
	}
	rectangle(img, cropRect, Scalar(0, 255, 0), 1, 8, 0);
	imshow(winName, img);
}
void onMouse(int event, int x, int y, int f, void*)
{
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		clicked = true;
		P1.x = x;
		P1.y = y;
		P2.x = x;
		P2.y = y;
		break;
	case EVENT_LBUTTONUP:
		clicked = false;
		P2.x = x;
		P2.y = y;
		break;
	case EVENT_MOUSEMOVE:
		if (clicked)
		{
			P2.x = x;
			P2.y = y;
		}
		break;
	default: break;
	}

	if (clicked)
	{
		if (P1.x > P2.x)
		{
			cropRect.x = P2.x;
			cropRect.width = P1.x - P2.x;
		}
		else
		{
			cropRect.x = P1.x;
			cropRect.width = P2.x - P1.x;
		}

		if (P1.y > P2.y)
		{
			cropRect.y = P2.y;
			cropRect.height= P1.y - P2.y;
		}
		else
		{
			cropRect.y = P1.y;
			cropRect.height = P2.y - P1.y;
		}
	}
	showImage();
}
int main(int argc, char** argv)
{
	cout << "드래그 & 드롭으로 이미지 잘라내기" << endl;
	cout << "S : 저장하기" << endl;
	cout << "R : ROI 초기화" << endl;
	cout << "Esc : 종료" << endl << endl;

	src = imread(argv[1], IMREAD_COLOR);
	namedWindow(winName, WINDOW_AUTOSIZE);
	setMouseCallback(winName, onMouse, NULL);
	imshow(winName, src);

	while (true)
	{
		char c = waitKey();
		if (c == 's' && !ROI.empty())
		{
			imwrite("croppedImage.jpg", ROI);
			cout << "Saved!" << endl;
		}
		if (c == 'r')
		{
			cropRect.x = 0;
			cropRect.y = 0;
			cropRect.width = 0;
			cropRect.height = 0;
		}
		if (c == 27) { break; }
		showImage();
	}
	return 0;
}