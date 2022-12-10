#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

// 벡터에서 중간값을 찾는 함수
double median(vector<double> vec) {

	int vecSize = vec.size();	// 벡터의 길이

	if (vecSize == 0) {	// 예외 처리
		throw domain_error("median of empty vector");
	}
	sort(vec.begin(), vec.end());	// 정렬

	int middle;
	double median;

	middle = vecSize / 2;
	if (vecSize % 2 == 0) { // 벡터 길이가 짝수인 경우 중간값 2개의 평균을 계산
		median = (vec[middle - 1] + vec[middle]) / 2;
	}
	else { // 홀수인 경우 중간값을 그대로 사용
		median = vec[middle];
	}
	return median;
}

int main(int argc, char** argv) {
	VideoCapture cap(argv[1]);
	
	if (!cap.isOpened()) {
		return -1;
	}
	Mat frame;
	int total_count = 0;

	vector< vector<double> > previousSlopePositiveLines;
	vector< vector<double> > previousSlopeNegativeLines;
	float previousPosSlopeMean;
	float previousNegSlopeMean;

	while (true)
	{
		cap >> frame;
		if (frame.empty())
		{
			printf("Incorrect video!");
			return -1;
		}

		Mat frameGray;	// 그레이스케일로 변환
		cvtColor(frame, frameGray, COLOR_BGR2GRAY);

		Mat frameGauss;	// 가우시안 스무딩
		GaussianBlur(frameGray, frameGauss, Size(9, 9), 0, 0);

		int minVal = 45;
		int maxVal = 100;

		Mat frameEdge;	// 엣지 탐지
		Canny(frameGauss, frameEdge, minVal, maxVal);

		// 관심영역 마스크 생성
		Mat roiMask(frame.size().height, frame.size().width, CV_8UC1, Scalar(0));
		Point p1 = Point(0, frame.size().height);					// 사다리꼴 좌표, 좌하단
		Point p2 = Point(550, 350);									// 좌상단
		Point p3 = Point(610, 350);									// 우상단
		Point p4 = Point(frame.size().width, frame.size().height);	// 우하단
		Point vertices1[] = { p1, p2, p3, p4 };
		vector<cv::Point> vertices(vertices1, vertices1 + sizeof(vertices1) / sizeof(Point));
		vector<vector<Point>> verticesToFill;
		verticesToFill.push_back(vertices);

		fillPoly(roiMask, verticesToFill, Scalar(255, 255, 255)); // 사다리꼴 영역만 하얀색으로 변환

		Mat frameMask = frameEdge.clone();	// 마스크를 원본에 적용
		bitwise_and(frameEdge, roiMask, frameMask);

		float rho = 2;
		float pi = 3.14159265358979323846;
		float theta = pi / 180;
		float threshold = 80;
		int minLineLength = 40;
		int maxLineGap = 100;

		vector<Vec4i> lines;	// 허프 직선 검색
		HoughLinesP(frameMask, lines, rho, theta, threshold, minLineLength, maxLineGap);
		
		if (!lines.empty() && lines.size() > 2)	{	// 직선이 1개 이상인지 확인
			Mat frameAllLines(	// 직선 이미지 초기화
				frame.size().height, 
				frame.size().width, 
				CV_8UC3, Scalar(0, 0, 0));

			for (size_t i = 0; i != lines.size(); ++i) {
				line(frameAllLines,	// 원본 이미지 위에 직선 그리기
					Point(lines[i][0], lines[i][1]),
					Point(lines[i][2], lines[i][3]), 
					Scalar(0, 0, 255), 3, 8);
			}

			// 검출된 직선을 좌/우 (Positive/Negative) 경사(Slope) 로 분리
			vector< vector<double> > slopePositiveLines; // format will be [x1 y1 x2 y2 slope]
			vector< vector<double> > slopeNegativeLines;
			vector<float> yValues;

			bool addedPos = false;
			bool addedNeg = false;
			int posCounter = 0;
			int negCounter = 0;
			
			// 모든 직선에 대한 처리
			for (size_t i = 0; i != lines.size(); ++i) {

				// 현재 직선의 좌표
				float x1 = lines[i][0];
				float y1 = lines[i][1];
				float x2 = lines[i][2];
				float y2 = lines[i][3];

				// 직선의 길이
				float lineLength = pow(pow(x2 - x1, 2) + pow(y2 - y1, 2), .5);

				// 직선의 길이가 충분한지 검사
				if (lineLength > 30) {

					// 0 으로 나누기 방지
					if (x2 != x1) {

						// 경사 계산
						float slope = (y2 - y1) / (x2 - x1);

						////////////////////////////////////////////////////////////////////////

						// 경사값이 양수
						if (slope > 0) {

							// x 축에 대해 직선의 각도 계산
							float tanTheta = tan((abs(y2 - y1)) / (abs(x2 - x1))); // tan(theta) value
							float angle = atan(tanTheta) * 180 / pi;

							// 세로 직각, 가로 직각의 선은 사용하지 않음
							if (abs(angle) < 85 && abs(angle) > 20) {

								// 행렬에 행을 추가
								slopeNegativeLines.resize(negCounter + 1);

								// 현재 행을 5 개의 열로 변환 [x1, y1, x2, y2, slope]
								slopeNegativeLines[negCounter].resize(5);

								// 값을 행에 추가
								slopeNegativeLines[negCounter][0] = x1;
								slopeNegativeLines[negCounter][1] = y1;
								slopeNegativeLines[negCounter][2] = x2;
								slopeNegativeLines[negCounter][3] = y2;
								slopeNegativeLines[negCounter][4] = -slope;

								// yValues 에 추가
								yValues.push_back(y1);
								yValues.push_back(y2);

								// 양수값을 갖는 경사임을 표시
								addedPos = true;

								// 반복 카운터
								negCounter++;
							}
						}

						////////////////////////////////////////////////////////////////////////

						// 경사값이 음수
						if (slope < 0) {

							// x 축에 대해 직선의 각도 계산
							float tanTheta = tan((abs(y2 - y1)) / (abs(x2 - x1))); // tan(theta) value
							float angle = atan(tanTheta) * 180 / pi;

							// 세로 직각, 가로 직각의 선은 사용하지 않음
							if (abs(angle) < 85 && abs(angle) > 20) {

								// 행렬에 행을 추가
								slopePositiveLines.resize(posCounter + 1);

								// 현재 행을 5 개의 열로 변환 [x1, y1, x2, y2, slope]
								slopePositiveLines[posCounter].resize(5);

								// 값을 행에 추가
								slopePositiveLines[posCounter][0] = x1;
								slopePositiveLines[posCounter][1] = y1;
								slopePositiveLines[posCounter][2] = x2;
								slopePositiveLines[posCounter][3] = y2;
								slopePositiveLines[posCounter][4] = -slope;

								// yValues 에 추가
								yValues.push_back(y1);
								yValues.push_back(y2);

								// 음수값을 갖는 경사임을 표시
								addedNeg = true;

								// 반복 카운터
								posCounter++;
							}
						}
					}
				}
			}

			////////////////////////////////////////////////////////////////////////

			// 양수값을 갖는 직선이 없으면 각도 기준을 조금 낮추어 다시 수행
			if (addedPos == false) {

				for (size_t i = 0; i != lines.size(); ++i) {

					float x1 = lines[i][0];
					float y1 = lines[i][1];
					float x2 = lines[i][2];
					float y2 = lines[i][3];

					float slope = (y2 - y1) / (x2 - x1);

					if (slope > 0 && x2 != x1) {

						float tanTheta = tan((abs(y2 - y1)) / (abs(x2 - x1)));
						float angle = atan(tanTheta) * 180 / pi;

						if (abs(angle) < 85 && abs(angle) > 15) {

							slopeNegativeLines.resize(negCounter + 1);

							slopeNegativeLines[negCounter].resize(5);

							slopeNegativeLines[negCounter][0] = x1;
							slopeNegativeLines[negCounter][1] = y1;
							slopeNegativeLines[negCounter][2] = x2;
							slopeNegativeLines[negCounter][3] = y2;
							slopeNegativeLines[negCounter][4] = -slope;

							yValues.push_back(y1);
							yValues.push_back(y2);

							addedPos = true;

							negCounter++;
						}
					}

				}
			}

			////////////////////////////////////////////////////////////////////////

			// 음수값을 갖는 직선이 없으면 각도 기준을 조금 낮추어 다시 수행
			if (addedNeg == false) {

				for (size_t i = 0; i != lines.size(); ++i) {

					float x1 = lines[i][0];
					float y1 = lines[i][1];
					float x2 = lines[i][2];
					float y2 = lines[i][3];

					float slope = (y2 - y1) / (x2 - x1);

					if (slope > 0 && x2 != x1) {

						float tanTheta = tan((abs(y2 - y1)) / (abs(x2 - x1)));
						float angle = atan(tanTheta) * 180 / pi;

						if (abs(angle) < 85 && abs(angle) > 15) {

							slopePositiveLines.resize(posCounter + 1);

							slopePositiveLines[posCounter].resize(5);

							slopePositiveLines[posCounter][0] = x1;
							slopePositiveLines[posCounter][1] = y1;
							slopePositiveLines[posCounter][2] = x2;
							slopePositiveLines[posCounter][3] = y2;
							slopePositiveLines[posCounter][4] = -slope;

							yValues.push_back(y1);
							yValues.push_back(y2);

							addedNeg = true;

							posCounter++;
						}
					}
				}
			}

			////////////////////////////////////////////////////////////////////////

			if (addedPos == false || addedNeg == false) {
				// 충분한 직선이 검출되지 않음
				cout << "Not enough lines found" << endl;
			}

			////////////////////////////////////////////////////////////////////////

			// 양의 경사값 (slope) 벡터 생성
			vector<double> positiveSlopes;
			for (unsigned int i = 0; i != slopePositiveLines.size(); ++i) {
				positiveSlopes.push_back(slopePositiveLines[i][4]);
			}

			// 양의 경사값 평균 계산
			double posSlopeMedian = median(positiveSlopes);

			vector<double> posSlopesGood;
			double posSum = 0.0; // good slope 합계

			// 유의미한 경사 (good slope) 만 남기는 작업 
			for (size_t i = 0; i != positiveSlopes.size(); ++i) {
				// 중간값과의 차이가 충분히 작은 경우 good slope 에 추가
				if (abs(positiveSlopes[i] - posSlopeMedian) < posSlopeMedian * .2) {
					posSlopesGood.push_back(positiveSlopes[i]);
					posSum += positiveSlopes[i];
				}
			}

			// good slope 의 평균 계산
			double posSlopeMean = posSum / posSlopesGood.size();
			
			// 평균값이 계산되지 않은 경우 (Nan) 유의미한 경사가 없는 것
			if (isnan(posSlopeMean)) {
				// 이 경우 이전 프레임의 경사 정보를 그대로 사용
				slopePositiveLines = previousSlopePositiveLines;
				posSlopeMean = previousPosSlopeMean;
			}

			////////////////////////////////////////////////////////////////////////

			// 음의 경사값 (slope) 벡터 생성
			vector<double> negativeSlopes;
			for (size_t i = 0; i != slopeNegativeLines.size(); ++i) {
				negativeSlopes.push_back(slopeNegativeLines[i][4]);
			}

			// 음의 경사값 평균 계산
			double negSlopeMedian = median(negativeSlopes);

			vector<double> negSlopesGood;
			double negSum = 0.0; // good slope 의 합계

			// 유의미한 경사 (good slope) 만 남기는 작업
			for (size_t i = 0; i != negativeSlopes.size(); ++i) {
				// 중간값과의 차이가 충분히 작은 경우 good slope 에 추가
				if (abs(negativeSlopes[i] - negSlopeMedian) < .9) {
					negSlopesGood.push_back(negativeSlopes[i]);
					negSum += negativeSlopes[i];
				}
			}

			// good slope 의 평균 계산
			double negSlopeMean = negSum / negSlopesGood.size();

			// 평균값이 계산되지 않은 경우 (Nan) 유의미한 경사가 없는 것
			if (isnan(negSlopeMean)) {
				// 이 경우 이전 프레임의 경사 정보를 그대로 사용
				slopeNegativeLines = previousSlopeNegativeLines;
				negSlopeMean = previousNegSlopeMean;
			}

			// 직선의 y 좌표가 0 일 때, x 좌표의 평균 계산 /////////////////////////////////////////////////
			
			vector<double> xInterceptPos; // 양의 직선에서 x 절편의 벡터
			for (size_t i = 0; i != slopePositiveLines.size(); ++i) {
				double x1 = slopePositiveLines[i][0];				// x 값
				double y1 = frame.rows - slopePositiveLines[i][1];	// y 값, y 축은 뒤집혀 있음
				double slope = slopePositiveLines[i][4];
				double yIntercept = y1 - slope * x1;				// y 절편
				double xIntercept = -yIntercept / slope;			// x 절편 계산, y = mx+b
				if (isnan(xIntercept) == 0) {						// nan 값이 아니면 추가
					xInterceptPos.push_back(xIntercept);
				}
			}

			// 양의 경사에서 x 절편의 중간값
			double xIntPosMed = median(xInterceptPos);

			// 유의미한 x 절편 값 계산
			vector<double> xIntPosGood;
			double xIntSum = 0.;

			// 평균값 기준으로 재계산
			for (size_t i = 0; i != slopePositiveLines.size(); ++i) {
				double x1 = slopePositiveLines[i][0];				// x 값
				double y1 = frame.rows - slopePositiveLines[i][1];	// y 값, y 축은 뒤집혀 있음
				double slope = slopePositiveLines[i][4];
				double yIntercept = y1 - slope * x1;				// y 절편
				double xIntercept = -yIntercept / slope;			// x 절편 계산, y = mx+b

				// x 절편이 nan 값이 아닐 것, 중간값에 충분히 근접할 것
				if (isnan(xIntercept) == 0 && abs(xIntercept - xIntPosMed) < .35 * xIntPosMed) {
					xIntPosGood.push_back(xIntercept); // add to 'good' vector
					xIntSum += xIntercept;
				}
			}

			// 양의 경사에서 x 절편의 평균
			double xInterceptPosMean = xIntSum / xIntPosGood.size();

			/////////////////////////////////////////////////////////////////

			vector<double> xInterceptNeg; // 음의 직선에서 x 절편의 벡터

			for (size_t i = 0; i != slopeNegativeLines.size(); ++i) {
				double x1 = slopeNegativeLines[i][0];				// x 값
				double y1 = frame.rows - slopeNegativeLines[i][1];	// y 값, y 축은 뒤집혀 있음
				double slope = slopeNegativeLines[i][4];
				double yIntercept = y1 - slope * x1;				// y 절편
				double xIntercept = -yIntercept / slope;			// x 절편 계산, y = mx+b
				if (isnan(xIntercept) == 0) {						// nan 값이 아니면 추가
					xInterceptNeg.push_back(xIntercept);
				}
			}

			// 음의 경사에서 x 절편의 중간값
			double xIntNegMed = median(xInterceptNeg);

			// 유의미한 x 절편 값 계산
			vector<double> xIntNegGood;
			double xIntSumNeg = 0.;

			// 평균값 기준으로 재계산
			for (size_t i = 0; i != slopeNegativeLines.size(); ++i) {
				double x1 = slopeNegativeLines[i][0];				// x 값
				double y1 = frame.rows - slopeNegativeLines[i][1];	// y 값, y 축은 뒤집혀 있음
				double slope = slopeNegativeLines[i][4];
				double yIntercept = y1 - slope * x1;				// y 절편
				double xIntercept = -yIntercept / slope;			// x 절편 계산, y = mx+b

				// x 절편이 nan 값이 아닐 것, 중간값에 충분히 근접할 것
				if (isnan(xIntercept) == 0 && abs(xIntercept - xIntNegMed) < .35 * xIntNegMed) {
					xIntNegGood.push_back(xIntercept);
					xIntSumNeg += xIntercept;
				}
			}

			// 음의 경사에서 x 절편의 평균
			double xInterceptNegMean = xIntSumNeg / xIntNegGood.size();

			// 차선 라인이 표시된 이미지 생성
			cv::Mat laneLineImage = frame.clone();
			cv::Mat laneFill = frame.clone();

			// 양의 경사
			float slope = posSlopeMean;
			double x1 = xInterceptPosMean;
			int y1 = 0;
			double y2 = frame.size().height - (frame.size().height - frame.size().height * .35);
			double x2 = (y2 - y1) / slope + x1;

			// 양의 경사를 이미지에 추가
			x1 = int(x1 + .5);
			x2 = int(x2 + .5);
			y1 = int(y1 + .5);
			y2 = int(y2 + .5);
			cv::line(laneLineImage, 
				cv::Point(x1, frame.size().height - y1), 
				cv::Point(x2, frame.size().height - y2),
				cv::Scalar(0, 255, 0), 3, 8);

			// 음의 경사
			slope = negSlopeMean;
			double x1N = xInterceptNegMean;
			int y1N = 0;
			double x2N = (y2 - y1N) / slope + x1N;

			// 음의 경사를 이미지에 추가
			x1N = int(x1N + .5);
			x2N = int(x2N + .5);
			y1N = int(y1N + .5);
			cv::line(laneLineImage, 
				cv::Point(x1N, frame.size().height - y1N), 
				cv::Point(x2N, frame.size().height - y2),
				cv::Scalar(0, 255, 0), 3, 8);

			// 좌표 변수
			cv::Point v1 = cv::Point(x1, frame.size().height - y1);
			cv::Point v2 = cv::Point(x2, frame.size().height - y2);
			cv::Point v3 = cv::Point(x1N, frame.size().height - y1N);
			cv::Point v4 = cv::Point(x2N, frame.size().height - y2);

			// 차선의 코너 지점
			cv::Point verticesBlend[] = { v1,v3,v4,v2 };
			std::vector<cv::Point> verticesVecBlend(verticesBlend, 
				verticesBlend + sizeof(verticesBlend) / sizeof(cv::Point));

			// fillPoly 에서 사용할 벡터
			std::vector<std::vector<cv::Point> > verticesfp;
			verticesfp.push_back(verticesVecBlend);

			// 영역에 색 칠하기
			cv::fillPoly(laneFill, verticesfp, cv::Scalar(0, 255, 255));

			// 이미지 블렌딩
			float opacity = .25;
			cv::Mat blendedIm;
			cv::addWeighted(laneFill, opacity, frame, 1 - opacity, 0, blendedIm);

			// 차선 라인 그리기
			cv::line(blendedIm, 
				cv::Point(x1, frame.size().height - y1), 
				cv::Point(x2, frame.size().height - y2),
				cv::Scalar(0, 255, 0), 8, 8);
			cv::line(blendedIm, 
				cv::Point(x1N, frame.size().height - y1N), 
				cv::Point(x2N, frame.size().height - y2),
				cv::Scalar(0, 255, 0), 8, 8);

			// 최종 결과 화면 표시
			cv::imshow("Lane Detection", blendedIm);
			total_count++;

			// 현재 프레임의 직선 정보를 저장
			previousSlopePositiveLines = slopePositiveLines;
			previousSlopeNegativeLines = slopeNegativeLines;
			previousPosSlopeMean = posSlopeMean;
			previousNegSlopeMean = negSlopeMean;
			
			if (waitKey(1) == 27) break; // ESC 입력 시 종료
		}
		else {
			cout << "Not enough lines found" << endl;
		}
	}
	return 0;
}