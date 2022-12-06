#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Function for finding median
double median(vector<double> vec) {

	// get size of vector
	int vecSize = vec.size();

	// if vector is empty throw error
	if (vecSize == 0) {
		throw domain_error("median of empty vector");
	}

	// sort vector
	sort(vec.begin(), vec.end());

	// define middle and median
	int middle;
	double median;

	// if even number of elements in vec, take average of two middle values
	if (vecSize % 2 == 0) {
		// a value representing the middle of the array. If array is of size 4 this is 2
		// if it's 8 then middle is 4
		middle = vecSize / 2;

		// take average of middle values, so if vector is [1, 2, 3, 4] we want average of 2 and 3
		// since we index at 0 middle will be the higher one vec[2] in the above vector is 3, and vec[1] is 2
		median = (vec[middle - 1] + vec[middle]) / 2;
	}

	// odd number of values in the vector
	else {
		middle = vecSize / 2; // take the middle again

		// if vector is 1 2 3 4 5, middle will be 5/2 = 2, and vec[2] = 3, the middle value
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
	while (true)
	{
		cap >> frame;
		if (frame.empty())
		{
			printf("Incorrect video!");
			return -1;
		}
		// imshow("Driving Video", frame);
		if (waitKey(1) == 27) break;

		// 그레이스케일로 변환
		Mat frameGray;
		cvtColor(frame, frameGray, COLOR_BGR2GRAY);
		
		/*imshow("Grayscale", frameGray);
		waitKey(0);*/

		// 가우시안 스무딩
		Mat frameGauss;
		GaussianBlur(frameGray, frameGauss, Size(9, 9), 0, 0);
		
		/*imshow("Gaussian blurred", frameGauss);
		waitKey(0);*/

		// 엣지 탐지
		int minVal = 45;
		int maxVal = 100;

		Mat frameEdge;
		Canny(frameGauss, frameEdge, minVal, maxVal);

		/*imshow("Edge", frameEdge);
		waitKey(0);*/

		// 관심영역 마스크 생성
		Mat roiMask(frame.size().height, frame.size().width, CV_8UC1, Scalar(0));
		Point p1 = Point(0, frame.size().height);	// 사다리꼴 좌표, 좌하단
		Point p2 = Point(550, 350);					// 좌상단
		Point p3 = Point(610, 350);					// 우상단
		Point p4 = Point(frame.size().width, frame.size().height);	// 우하단
		Point vertices1[] = { p1, p2, p3, p4 };
		vector<cv::Point> vertices(vertices1, vertices1 + sizeof(vertices1) / sizeof(Point));
		vector<vector<Point>> verticesToFill;
		verticesToFill.push_back(vertices);

		fillPoly(roiMask, verticesToFill, Scalar(255, 255, 255)); // 사다리꼴 영역만 하얀색으로 변환

		/*imshow("Mask", roiMask);
		waitKey(0);*/

		// 마스크를 원본에 적용
		Mat frameMask = frameEdge.clone();
		bitwise_and(frameEdge, roiMask, frameMask);

		/*imshow("Mask applied", frameMask);
		waitKey(0);*/

		// 허프 직선
		float rho = 2;
		float pi = 3.14159265358979323846;
		float theta = pi / 180;
		float threshold = 80;
		int minLineLength = 40;
		int maxLineGap = 100;

		vector<Vec4i> lines;
		HoughLinesP(frameMask, lines, rho, theta, threshold, minLineLength, maxLineGap);

		// 라인이 1개 이상인지 확인
		if (!lines.empty() && lines.size() > 2)
		{
			// 라인 이미지 초기화
			Mat frameAllLines(frame.size().height, frame.size().width, CV_8UC3, Scalar(0, 0, 0));

			for (size_t i = 0; i != lines.size(); ++i) {
				// 원본 이미지 위에 라인 그리기
				line(frameAllLines,
					Point(lines[i][0], lines[i][1]),
					Point(lines[i][2], lines[i][3]), 
					Scalar(0, 0, 255), 3, 8);
			}

			/*imshow("Hough Lines", frameAllLines);
			waitKey(0);*/

			// 검출된 라인을 좌/우 (Positive/Negative) 경사(Slope) 로 분리
			vector< vector<double> > slopePositiveLines; // format will be [x1 y1 x2 y2 slope]
			vector< vector<double> > slopeNegativeLines;
			vector<float> yValues;

			bool addedPos = false;
			bool addedNeg = false;
			int posCounter = 0;
			int negCounter = 0;
			
			// Loop through all lines
			for (size_t i = 0; i != lines.size(); ++i) {

				// Get points for current line
				float x1 = lines[i][0];
				float y1 = lines[i][1];
				float x2 = lines[i][2];
				float y2 = lines[i][3];

				// get line length
				float lineLength = pow(pow(x2 - x1, 2) + pow(y2 - y1, 2), .5);

				// if line is long enough
				if (lineLength > 30) {

					// dont divide by zero
					if (x2 != x1) {

						// get slope
						float slope = (y2 - y1) / (x2 - x1);

						// Check if slope is positive
						if (slope > 0) {

							// Find angle of line wrt x axis.
							float tanTheta = tan((abs(y2 - y1)) / (abs(x2 - x1))); // tan(theta) value
							float angle = atan(tanTheta) * 180 / pi;

							// Only pass good line angles,  dont want verticalish/horizontalish lines
							if (abs(angle) < 85 && abs(angle) > 20) {

								// Add a row to the matrix
								slopeNegativeLines.resize(negCounter + 1);

								// Reshape current row to 5 columns [x1, y1, x2, y2, slope]
								slopeNegativeLines[negCounter].resize(5);

								// Add values to row
								slopeNegativeLines[negCounter][0] = x1;
								slopeNegativeLines[negCounter][1] = y1;
								slopeNegativeLines[negCounter][2] = x2;
								slopeNegativeLines[negCounter][3] = y2;
								slopeNegativeLines[negCounter][4] = -slope;

								// add yValues
								yValues.push_back(y1);
								yValues.push_back(y2);

								// Note that we added a positive slope line
								addedPos = true;

								// iterate the counter
								negCounter++;

							}

						}

						// Check if slope is Negative
						if (slope < 0) {

							// Find angle of line wrt x axis.
							float tanTheta = tan((abs(y2 - y1)) / (abs(x2 - x1))); // tan(theta) value
							float angle = atan(tanTheta) * 180 / pi;

							// Only pass good line angles,  dont want verticalish/horizontalish lines
							if (abs(angle) < 85 && abs(angle) > 20) {

								// Add a row to the matrix
								slopePositiveLines.resize(posCounter + 1);

								// Reshape current row to 5 columns [x1, y1, x2, y2, slope]
								slopePositiveLines[posCounter].resize(5);

								// Add values to row
								slopePositiveLines[posCounter][0] = x1;
								slopePositiveLines[posCounter][1] = y1;
								slopePositiveLines[posCounter][2] = x2;
								slopePositiveLines[posCounter][3] = y2;
								slopePositiveLines[posCounter][4] = -slope;

								// add yValues
								yValues.push_back(y1);
								yValues.push_back(y2);

								// Note that we added a positive slope line
								addedNeg = true;

								// iterate counter
								posCounter++;

							}
						}	// if slope < 0
					}	// if x2 != x1
				}	// if lineLength > 30
			}	// looping though all lines

			// If we didn't get any positive lines, go though again and just add any positive slope lines
			// Be less strict
			if (addedPos == false) { // if we didnt add any positive lines

				// loop through lines
				for (size_t i = 0; i != lines.size(); ++i) {

					// Get points for current line
					float x1 = lines[i][0];
					float y1 = lines[i][1];
					float x2 = lines[i][2];
					float y2 = lines[i][3];

					// Get slope
					float slope = (y2 - y1) / (x2 - x1);

					// Check if slope is positive
					if (slope > 0 && x2 != x1) {

						// Find angle of line wrt x axis.
						float tanTheta = tan((abs(y2 - y1)) / (abs(x2 - x1))); // tan(theta) value
						float angle = atan(tanTheta) * 180 / pi;

						// Only pass good line angles,  dont want verticalish/horizontalish lines
						if (abs(angle) < 85 && abs(angle) > 15) {

							// Add a row to the matrix
							slopeNegativeLines.resize(negCounter + 1);

							// Reshape current row to 5 columns [x1, y1, x2, y2, slope]
							slopeNegativeLines[negCounter].resize(5);

							// Add values to row
							slopeNegativeLines[negCounter][0] = x1;
							slopeNegativeLines[negCounter][1] = y1;
							slopeNegativeLines[negCounter][2] = x2;
							slopeNegativeLines[negCounter][3] = y2;
							slopeNegativeLines[negCounter][4] = -slope;

							// add yValues
							yValues.push_back(y1);
							yValues.push_back(y2);

							// Note that we added a positive slope line
							addedPos = true;

							// iterate the counter
							negCounter++;
						}
					}

				}
			} // if addedPos == false

			// If we didn't get any negative lines, go though again and just add any positive slope lines
			// Be less strict
			if (addedNeg == false) { // if we didnt add any positive lines

				// loop through lines
				for (size_t i = 0; i != lines.size(); ++i) {

					// Get points for current line
					float x1 = lines[i][0];
					float y1 = lines[i][1];
					float x2 = lines[i][2];
					float y2 = lines[i][3];

					// Get slope
					float slope = (y2 - y1) / (x2 - x1);

					// Check if slope is positive
					if (slope > 0 && x2 != x1) {

						// Find angle of line wrt x axis.
						float tanTheta = tan((abs(y2 - y1)) / (abs(x2 - x1))); // tan(theta) value
						float angle = atan(tanTheta) * 180 / pi;

						// Only pass good line angles,  dont want verticalish/horizontalish lines
						if (abs(angle) < 85 && abs(angle) > 15) {

							// Add a row to the matrix
							slopePositiveLines.resize(posCounter + 1);

							// Reshape current row to 5 columns [x1, y1, x2, y2, slope]
							slopePositiveLines[posCounter].resize(5);

							// Add values to row
							slopeNegativeLines[posCounter][0] = x1;
							slopeNegativeLines[posCounter][1] = y1;
							slopeNegativeLines[posCounter][2] = x2;
							slopeNegativeLines[posCounter][3] = y2;
							slopeNegativeLines[posCounter][4] = -slope;

							// add yValues
							yValues.push_back(y1);
							yValues.push_back(y2);

							// Note that we added a positive slope line
							addedNeg = true;

							// iterate the counter
							posCounter++;
						}
					}

				}
			} // if addedNeg == false

			// If we still dont have lines then fuck
			if (addedPos == false || addedNeg == false) {
				cout << "Not enough lines found" << endl;
			}


			//-----------------GET POSITIVE/NEGATIVE SLOPE AVERAGES-----------------------
			// Average the position of lines and extrapolate to the top and bottom of the lane.

			// Add positive slopes from slopePositiveLines into a vector positive slopes
			vector<float> positiveSlopes;
			for (unsigned int i = 0; i != slopePositiveLines.size(); ++i) {
				positiveSlopes.push_back(slopePositiveLines[i][4]);
			}

			// Get median of positiveSlopes
			sort(positiveSlopes.begin(), positiveSlopes.end()); // sort vec
			int middle; // define middle value
			double posSlopeMedian; // define positive slope median

			// if even number of elements in vec, take average of two middle values
			if (positiveSlopes.size() % 2 == 0) {

				// a value representing the middle of the array. If array is of size 4 this is 2
				// if it's 8 then middle is 4
				middle = positiveSlopes.size() / 2;

				// take average of middle values, so if vector is [1, 2, 3, 4] we want average of 2 and 3
				// since we index at 0 middle will be the higher one vec[2] in the above vector is 3, and vec[1] is 2
				posSlopeMedian = (positiveSlopes[middle - 1] + positiveSlopes[middle]) / 2;
			}

			// odd number of values in the vector
			else {
				middle = positiveSlopes.size() / 2; // take the middle again

				// if vector is 1 2 3 4 5, middle will be 5/2 = 2, and vec[2] = 3, the middle value
				posSlopeMedian = positiveSlopes[middle];
			}

			// Define vector of 'good' slopes, slopes that are drastically different than the others are thrown out
			vector<float> posSlopesGood;
			float posSum = 0.0; // sum so we'll be able to get mean

			// Loop through positive slopes and add the good ones
			for (size_t i = 0; i != positiveSlopes.size(); ++i) {

				// check difference between current slope and the median. If the difference is small enough it's good
				if (abs(positiveSlopes[i] - posSlopeMedian) < posSlopeMedian * .2) {
					posSlopesGood.push_back(positiveSlopes[i]); // Add slope to posSlopesGood
					posSum += positiveSlopes[i]; // add to sum
				}
			}

			// Get mean of good positive slopes
			float posSlopeMean = posSum / posSlopesGood.size();

			////////////////////////////////////////////////////////////////////////

			// Add negative slopes from slopeNegativeLines into a vector negative slopes
			vector<float> negativeSlopes;
			for (size_t i = 0; i != slopeNegativeLines.size(); ++i) {
				negativeSlopes.push_back(slopeNegativeLines[i][4]);
			}

			// Get median of negativeSlopes
			sort(negativeSlopes.begin(), negativeSlopes.end()); // sort vec
			int middleNeg; // define middle value
			double negSlopeMedian; // define negative slope median

			// if even number of elements in vec, take average of two middle values
			if (negativeSlopes.size() % 2 == 0) {

				// a value representing the middle of the array. If array is of size 4 this is 2
				// if it's 8 then middle is 4
				middleNeg = negativeSlopes.size() / 2;

				// take average of middle values, so if vector is [1, 2, 3, 4] we want average of 2 and 3
				// since we index at 0 middle will be the higher one vec[2] in the above vector is 3, and vec[1] is 2
				negSlopeMedian = (negativeSlopes[middleNeg - 1] + negativeSlopes[middleNeg]) / 2;
			}

			// odd number of values in the vector
			else {
				middleNeg = negativeSlopes.size() / 2; // take the middle again

				// if vector is 1 2 3 4 5, middle will be 5/2 = 2, and vec[2] = 3, the middle value
				negSlopeMedian = negativeSlopes[middle];
			}

			// Define vector of 'good' slopes, slopes that are drastically different than the others are thrown out
			vector<float> negSlopesGood;
			float negSum = 0.0; // sum so we'll be able to get mean

			//std::cout << "negativeSlopes.size(): " << negativeSlopes.size() << endl;
			//std::cout << "condition: " << negSlopeMedian*.2 << endl;

			// Loop through positive slopes and add the good ones
			for (size_t i = 0; i != negativeSlopes.size(); ++i) {

				//cout << "check: " << negativeSlopes[i]  << endl;

				// check difference between current slope and the median. If the difference is small enough it's good
				if (abs(negativeSlopes[i] - negSlopeMedian) < .9) { // < negSlopeMedian*.2
					negSlopesGood.push_back(negativeSlopes[i]); // Add slope to negSlopesGood
					negSum += negativeSlopes[i]; // add to sum
				}
			}

			//cout << endl;
			// Get mean of good positive slopes
			float negSlopeMean = negSum / negSlopesGood.size();
			//std::cout << "negSum: " << negSum << " negSlopesGood.size(): " << negSlopesGood.size() << " negSlopeMean: " << negSlopeMean << endl;


			//----------------GET AVERAGE X COORD WHEN Y COORD OF LINE = 0--------------------
			// Positive Lines
			vector<double> xInterceptPos; // define vector for x intercepts of positive slope lines

			// Loop through positive slope lines, find and store x intercept values
			for (size_t i = 0; i != slopePositiveLines.size(); ++i) {
				double x1 = slopePositiveLines[i][0]; // x value
				double y1 = frame.rows - slopePositiveLines[i][1]; // y value...yaxis is flipped
				double slope = slopePositiveLines[i][4];
				double yIntercept = y1 - slope * x1; // yintercept of line
				double xIntercept = -yIntercept / slope; // find x intercept based off y = mx+b
				if (isnan(xIntercept) == 0) { // check for nan
					xInterceptPos.push_back(xIntercept); // add value
				}
			}

			// Get median of x intercepts for positive slope lines
			double xIntPosMed = median(xInterceptPos);

			// Define vector storing 'good' x intercept values, same concept as the slope calculations before
			vector<double> xIntPosGood;
			double xIntSum = 0.; // for finding avg

			// Now that we got median, loop through lines again and compare values against median
			for (size_t i = 0; i != slopePositiveLines.size(); ++i) {
				double x1 = slopePositiveLines[i][0]; // x value
				double y1 = frame.rows - slopePositiveLines[i][1]; // y value...yaxis is flipped
				double slope = slopePositiveLines[i][4];
				double yIntercept = y1 - slope * x1; // yintercept of line
				double xIntercept = -yIntercept / slope; // find x intercept based off y = mx+b

				// check for nan and check if it's close enough to the median
				if (isnan(xIntercept) == 0 && abs(xIntercept - xIntPosMed) < .35 * xIntPosMed) {
					xIntPosGood.push_back(xIntercept); // add to 'good' vector
					xIntSum += xIntercept;
				}
			}

			// Get mean x intercept value for positive slope lines
			double xInterceptPosMean = xIntSum / xIntPosGood.size();

			/////////////////////////////////////////////////////////////////
			// Negative Lines
			vector<double> xInterceptNeg; // define vector for x intercepts of negative slope lines

			// Loop through negative slope lines, find and store x intercept values
			for (size_t i = 0; i != slopeNegativeLines.size(); ++i) {
				double x1 = slopeNegativeLines[i][0]; // x value
				double y1 = frame.rows - slopeNegativeLines[i][1]; // y value...yaxis is flipped
				double slope = slopeNegativeLines[i][4];
				double yIntercept = y1 - slope * x1; // yintercept of line
				double xIntercept = -yIntercept / slope; // find x intercept based off y = mx+b
				if (isnan(xIntercept) == 0) { // check for nan
					xInterceptNeg.push_back(xIntercept); // add value
				}
			}

			// Get median of x intercepts for negative slope lines
			double xIntNegMed = median(xInterceptNeg);

			// Define vector storing 'good' x intercept values, same concept as the slope calculations before
			vector<double> xIntNegGood;
			double xIntSumNeg = 0.; // for finding avg

			// Now that we got median, loop through lines again and compare values against median
			for (size_t i = 0; i != slopeNegativeLines.size(); ++i) {
				double x1 = slopeNegativeLines[i][0]; // x value
				double y1 = frame.rows - slopeNegativeLines[i][1]; // y value...yaxis is flipped
				double slope = slopeNegativeLines[i][4];
				double yIntercept = y1 - slope * x1; // yintercept of line
				double xIntercept = -yIntercept / slope; // find x intercept based off y = mx+b

				// check for nan and check if it's close enough to the median
				if (isnan(xIntercept) == 0 && abs(xIntercept - xIntNegMed) < .35 * xIntNegMed) {
					xIntNegGood.push_back(xIntercept); // add to 'good' vector
					xIntSumNeg += xIntercept;
				}
			}

			// Get mean x intercept value for negative slope lines
			double xInterceptNegMean = xIntSumNeg / xIntNegGood.size();
			//gotLines = true;


			//-----------------------PLOT LANE LINES------------------------
			// Need end points of line to draw in. Have x1,y1 (xIntercept,im.shape[1]) where
			// im.shape[1] is the bottom of the image. take y2 as some num (min/max y in the good lines?)
			// then find corresponding x

			// Create image, lane lines on real image
			cv::Mat laneLineImage = frame.clone();
			cv::Mat laneFill = frame.clone();

			// Positive Slope Line
			float slope = posSlopeMean;
			double x1 = xInterceptPosMean;
			int y1 = 0;
			double y2 = frame.size().height - (frame.size().height - frame.size().height * .35);
			double x2 = (y2 - y1) / slope + x1;

			// Add positive slope line to image
			x1 = int(x1 + .5);
			x2 = int(x2 + .5);
			y1 = int(y1 + .5);
			y2 = int(y2 + .5);
			cv::line(laneLineImage, cv::Point(x1, frame.size().height - y1), cv::Point(x2, frame.size().height - y2),
				cv::Scalar(0, 255, 0), 3, 8);


			// Negative Slope Line
			slope = negSlopeMean;
			double x1N = xInterceptNegMean;
			int y1N = 0;
			double x2N = (y2 - y1N) / slope + x1N;

			// Add negative slope line to image
			x1N = int(x1N + .5);
			x2N = int(x2N + .5);
			y1N = int(y1N + .5);
			cv::line(laneLineImage, cv::Point(x1N, frame.size().height - y1N), cv::Point(x2N, frame.size().height - y2),
				cv::Scalar(0, 255, 0), 3, 8);

			// Plot positive and negative lane lines
			//cv::imshow("Lane lines on image", laneLineImage);
			//cv::waitKey(0); // wait for a key press


			// -----------------BLEND IMAGE-----------------------
			// Use cv::Point type for x,y points
			cv::Point v1 = cv::Point(x1, frame.size().height - y1);
			cv::Point v2 = cv::Point(x2, frame.size().height - y2);
			cv::Point v3 = cv::Point(x1N, frame.size().height - y1N);
			cv::Point v4 = cv::Point(x2N, frame.size().height - y2);

			// create vector from array of corner points of lane
			cv::Point verticesBlend[] = { v1,v3,v4,v2 };
			std::vector<cv::Point> verticesVecBlend(verticesBlend, verticesBlend + sizeof(verticesBlend) / sizeof(cv::Point));

			// Create vector of vectors to be used in fillPoly, add the vertices we defined above
			std::vector<std::vector<cv::Point> > verticesfp;
			verticesfp.push_back(verticesVecBlend);

			// Fill area created from vector points
			cv::fillPoly(laneFill, verticesfp, cv::Scalar(0, 255, 255));

			// Blend image
			float opacity = .25;
			cv::Mat blendedIm;
			cv::addWeighted(laneFill, opacity, frame, 1 - opacity, 0, blendedIm);

			// Plot lane lines
			cv::line(blendedIm, cv::Point(x1, frame.size().height - y1), cv::Point(x2, frame.size().height - y2),
				cv::Scalar(0, 255, 0), 8, 8);
			cv::line(blendedIm, cv::Point(x1N, frame.size().height - y1N), cv::Point(x2N, frame.size().height - y2),
				cv::Scalar(0, 255, 0), 8, 8);

			// Show final frame
			cv::imshow("Final Output", blendedIm);
			// cv::waitKey(0);
		} // end if we got more than one line
		else // We do none of that if we don't see enough lines
		{
			cout << "Not enough lines found" << endl;
		}
	}
	return 0;
}