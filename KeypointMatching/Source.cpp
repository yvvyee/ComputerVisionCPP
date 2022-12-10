#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char** argv) {
	if (argc != 3)	{
		return -1;
	}

	Mat img_1 = imread(argv[1], IMREAD_GRAYSCALE);
	Mat img_2 = imread(argv[2], IMREAD_GRAYSCALE);

	if (!img_1.data || !img_2.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return -1;
	}

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	Ptr<SURF> detector = SURF::create(minHessian);// detector(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	//-- Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;

	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1),
		DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1),
		DrawMatchesFlags::DEFAULT);

	//-- Show detected (drawn) keypoints
	imshow("Keypoints 1", img_keypoints_1);
	imshow("Keypoints 2", img_keypoints_2);

	 // Construction of the SURF descriptor extractor 
	Ptr<SURF> extractor = SURF::create();

	// Extraction of the SURF descriptors
	cv::Mat descriptors1, descriptors2;
	extractor->compute(img_1, keypoints_1, descriptors1);
	extractor->compute(img_2, keypoints_2, descriptors2);
	std::cout << "descriptor matrix size: " << descriptors1.rows << " by "
		<< descriptors1.cols << std::endl;
	// Construction of the matcher 
	cv::BFMatcher matcher(cv::NORM_L2, false);

	// Match the two image descriptors
	std::vector<cv::DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);
	std::cout << "Number of matched points: " << matches.size() << std::endl;

	//-- Draw matches---//
	cv::Mat img_matches1;
	cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches1);

	//-- Show detected matches
	imshow("Matches", img_matches1);

	//--- Filtering loop ---//
	int limit = 100; // 매칭 포인트 수 제한
	std::nth_element(matches.begin(),    // initial position
		matches.begin() + limit - 1, // position of the sorted element
		matches.end());     // end position
	// remove all elements after the 25th
	matches.erase(matches.begin() + limit, matches.end());

	cv::Mat imageMatches2;
	cv::drawMatches(img_1, keypoints_1,  // 1st image and its keypoints
		img_2, keypoints_2,  // 2nd image and its keypoints
		matches,// the matches
		imageMatches2); // color of the lines
	cv::namedWindow("Filtered Matches");
	cv::imshow("Filtered Matches", imageMatches2);

	waitKey(0); return 0;

}