#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// salt-pepper 노이즈 실습

void SaltPepper(Mat& img, int n)
{
    for (int k = 0; k < n; k++) {
        int i = rand() % img.cols;
        int j = rand() % img.rows;

        if (img.channels() == 1) 
        {	// Gray scale image
            img.at<uchar>(j, i) = 255;
        }
        else if (img.channels() == 3) 
        {	// Color image
            img.at<Vec3b>(j, i)[0] = 255;
            img.at<Vec3b>(j, i)[1] = 255;
            img.at<Vec3b>(j, i)[2] = 255;
        }
    }
}

int main(int argc, char** argv)
{
    // 사용자로부터 이미지 파일 경로를 입력받고 객체 생성 
    Mat src = imread(argv[1], IMREAD_COLOR);
    if (src.empty()) 
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    Mat dst = src.clone();
    SaltPepper(dst, 30000);


    // 윈도우 생성 
    namedWindow("원본 이미지", 1);
    namedWindow("변환 이미지", 1);

    // 출력
    imshow("원본 이미지", src);
    imshow("변환 이미지", dst);

    // 대기
    waitKey();
    return 0;
}