#include <opencv2/opencv.hpp>
using namespace cv;

// 감마 보정 실습

int main(int argc, char** argv) 
{
    // 사용자로부터 이미지 파일 경로를 입력받고 객체 생성 
    Mat src = imread(argv[1]);
    Mat dst = Mat(src.rows, src.cols, CV_8UC1);

    cvtColor(src, src, COLOR_BGR2GRAY);
    dst = Scalar(0);

    std::cout << " Basic Linear Transforms: Gamma correction " << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << " gamma 입력 ( 1.0 ~ 3.0 ) : ";

    double gamma = 0.;
    std::cin >> gamma;

    int nl = src.rows;    // 이미지의 세로 길이 (행)
    int nc = src.cols * src.channels();// 각 행의 데이터 수

    for (int j = 0; j < nl; j++) {

        //-- j열의 주소 (nc 개만큼­) 가져오기 --//
        uchar* pSrc = src.ptr<uchar>(j);
        uchar* pDst = dst.ptr<uchar>(j);

        for (int i = 0; i < nc; i++)
        {
            int pixelValue = (int)pSrc[i];
            pDst[i] = pow(pixelValue, gamma);

        }
    }
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