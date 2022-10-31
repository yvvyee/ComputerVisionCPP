#include <opencv2/opencv.hpp>
using namespace cv;

// 영상 색상 변경 실습 (직선의 방정식 y = ax + b)
double alpha;	// 기울기
int beta;		// 절편

int main(int argc, char** argv) 
{
	// 사용자로부터 이미지 파일 경로를 입력받고 객체 생성 
	Mat image = imread(argv[1]);
	Mat new_image = Mat::zeros(image.size(), image.type());

	/// Initialize values 
	std::cout << " Basic Linear Transforms " << std::endl;
	std::cout << "-------------------------" << std::endl;
	std::cout << " alpha 입력 ( 1.0 ~ 3.0 ): ";
	std::cin >> alpha;
	if ((alpha < 1.) || (alpha > 3.))
	{
		std::cout << "\'alpha\' 입력 오류" << std::endl;
		return -1;
	}

	std::cout << " beta 입력 ( 0 ~ 100 ) : ";
	std::cin >> beta;
	if ((beta < 0) || (beta > 100))
	{
		std::cout << "\'beta\' 입력 오류" << std::endl;
		return -1;
	}

	/// new_image(i,j) = alpha * image(i,j) + beta 
	for (int i = 0; i < image.rows; i++) 
	{
		for (int j = 0; j < image.cols; j++) 
		{
			for (int c = 0; c < 3; c++) 
			{
				new_image.at<Vec3b>(i, j)[c] 
					= saturate_cast<uchar>(
						alpha * (image.at<Vec3b>(i, j)[c]) + beta);
			}
		}
	}
	// 윈도우 생성 
	namedWindow("원본 이미지", 1);
	namedWindow("변환 이미지", 1);

	// 출력
	imshow("원본 이미지", image);
	imshow("변환 이미지", new_image);

	// 대기
	waitKey();
	return 0;
}