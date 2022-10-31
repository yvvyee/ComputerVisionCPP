#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// Posterization (Quantization 또는 Color reduction) 실습

void ColorReduceV1(Mat& image, int div) 
{
    // 256 * 256 * 256 = 16,777,216
    // (256 / div) * (256 / div) * (256 / div)

    Mat_<Vec3b>::iterator it_cur = image.begin<Vec3b>();
    Mat_<Vec3b>::iterator it_end = image.end<Vec3b>();

    for (; it_cur != it_end; ++it_cur) 
    {
        (*it_cur)[0] = (*it_cur)[0] / div * div + div / 2;
        (*it_cur)[1] = (*it_cur)[1] / div * div + div / 2;
        (*it_cur)[2] = (*it_cur)[2] / div * div + div / 2;
    }
}

void ColorReduceV2(Mat& image, int div)
{

}

void ColorReduceV3(Mat& image, int div)
{

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

    Mat dst1 = src.clone(); // 원본을 결과 버퍼에 복사
    Mat dst2 = src.clone();
    Mat dst3 = src.clone();

    // 버전 1
    double t1 = (double)getTickCount();
    ColorReduceV1(dst1, 150);
    t1 = ((double)getTickCount() - t1) / getTickFrequency();

    // 버전 2
    double t2 = (double)getTickCount();
    ColorReduceV2(dst2, 128);
    t2 = ((double)getTickCount() - t1) / getTickFrequency();

    // 버전 3
    double t3 = (double)getTickCount();
    ColorReduceV3(dst3, 128);
    t3 = ((double)getTickCount() - t1) / getTickFrequency();

    // 시간 출력
    cout << "수행시간1: " << t1 << endl;
    cout << "수행시간2: " << t2 << endl;
    cout << "수행시간3: " << t3 << endl;

    // 윈도우 생성 
    namedWindow("원본 이미지", 1);
    namedWindow("변환 이미지", 1);

    // 출력
    imshow("원본 이미지", src);
    imshow("변환 이미지", dst1);

    // 대기
    waitKey();
    return 0;
}