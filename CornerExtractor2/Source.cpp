#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	if (argc != 2) {
		return -1;
	}

	Mat img_object = imread(argv[1], IMREAD_COLOR);
	Mat img_gray;
	cvtColor(img_object, img_gray, COLOR_BGR2GRAY);

	// FAST 기반 키포인트 추출
	Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(50, true);

	// Compute keypoints and descriptor from the source image in advance
	vector<KeyPoint> keypoints;
	Mat descriptors;

	// detecting and computing keypoints and descriptors
	detector->detect(img_gray, keypoints);
	printf(" ==> original image:%d keypoints are found.\n", (int)keypoints.size());

	for (int i = 0; i < keypoints.size(); i++) {
		KeyPoint kp = keypoints[i];
		circle(img_object, kp.pt, cvRound(kp.size * 0.25), Scalar(255, 255, 0), 1, 8, 0);
	}

	namedWindow("FAST Keypoints");
	imshow("FAST Keypoints", img_object);

	waitKey(0);
	return 0;
}
