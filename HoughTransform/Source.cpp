#include <opencv2/opencv.hpp>
using namespace cv;

float const PI = 3.14;

int main(int argc, char** argv)
{
	cv::Mat image = cv::imread(argv[1], IMREAD_GRAYSCALE);
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", image);

	// 캐니 알고리즘 적용
	cv::Mat contours;
	cv::Canny(image, contours, 125, 350);

	// 선 감지 위한 허프 변환
	std::vector<cv::Vec2f> lines;
	cv::HoughLines(contours, lines,
		1, PI / 180, // 단계별 크기
	80);  // 투표(vote) 최대 개수 ? 수에 따른 변화 관찰 필요 60, 40 등

	// 선 그리기
	cv::Mat result(contours.rows, contours.cols, CV_8U, cv::Scalar(255));
	std::cout << "Lines detected: " << lines.size() << std::endl;

	// 선 벡터를 반복해 선 그리기
	std::vector<cv::Vec2f>::const_iterator it = lines.begin();
	while (it != lines.end()) {
		float rho = (*it)[0];   // 첫 번째 요소는 rho 거리
		float theta = (*it)[1]; // 두 번째 요소는 델타 각도
		// 수직 행
		if (theta < PI / 4. || theta > 3. * PI / 4.) {
			// 첫 행에서 해당 선의 교차점
			cv::Point pt1(rho / cos(theta), 0);
			// 마지막 행에서 해당 선의 교차점
			cv::Point pt2((rho - result.rows * sin(theta)) 
				/ cos(theta), result.rows);
			// 하얀 선으로 그리기
			cv::line(image, pt1, pt2, cv::Scalar(255), 1); 
		}
		// 수평 행
		else {
			// 첫 번째 열에서 해당 선의 교차점  
			cv::Point pt1(0, rho / sin(theta)); 
			// 마지막 열에서 해당 선의 교차점
			cv::Point pt2(result.cols, (rho - result.cols 
				* cos(theta)) / sin(theta));
			// 하얀 선으로 그리기
			cv::line(image, pt1, pt2, cv::Scalar(255), 1); 
		}
		std::cout << "line: (" << rho << "," << theta << ")\n";
		++it;
	}
	cv::namedWindow("Detected Lines with Hough");
	cv::imshow("Detected Lines with Hough", image);
	cv::waitKey(0);
	return 0;
}
