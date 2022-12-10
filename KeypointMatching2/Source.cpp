#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main(int argc, char** argv) {

    // image read
    Mat img1 = cv::imread(argv[1], IMREAD_COLOR);
    Mat img2 = cv::imread(argv[2], IMREAD_COLOR);
    

    if (!img1.data || !img2.data) {
        std::cout << " --(!) Error reading images " << std::endl;
        return -1;
    }

    // SIFT feature detector and feature extractor
    cv::Ptr<SIFT> sift;
    sift = SIFT::create(0, 4, 0.04, 10, 1.6);

    // Compute keypoints and descriptor from the source image in advance
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;


    sift->detect(img1, keypoints1);
    sift->compute(img1, keypoints1, descriptors1);

    printf("original image:%d keypoints are found.\n", (int)keypoints1.size());

    for (int i = 0; i < keypoints1.size(); i++) {
        KeyPoint kp = keypoints1[i];
        circle(img1, kp.pt, cvRound(kp.size * 0.25), Scalar(255, 255, 0), 1, 8, 0);
    }

    namedWindow("SIFT Keypoints-src");
    imshow("SIFT Keypoints-src", img1);

    sift->detect(img2, keypoints2);
    sift->compute(img2, keypoints2, descriptors2);

    printf("original image:%d keypoints are found.\n", (int)keypoints2.size());

    for (int i = 0; i < keypoints2.size(); i++) {
        KeyPoint kp = keypoints2[i];
        circle(img2, kp.pt, cvRound(kp.size * 0.25), Scalar(255, 255, 0), 1, 8, 0);
    }

    namedWindow("SIFT Keypoints-tgt");
    imshow("SIFT Keypoints-tgt", img2);

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match(descriptors1, descriptors2, matches);

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < descriptors1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;

    for (int i = 0; i < descriptors1.rows; i++)
    {
        if (matches[i].distance < 3 * min_dist) {
            good_matches.push_back(matches[i]);
        }
    }

    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2,
        good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
        std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imshow("Matched Image", img_matches);

    //--- Filtering loop ---//
    int limit = 50;
    std::nth_element(matches.begin(),    // initial position
        matches.begin() + limit - 1, // position of the sorted element
        matches.end());     // end position
    // remove all elements after the 31th
    matches.erase(matches.begin() + limit, matches.end());

    cv::Mat imageMatches2;
    cv::drawMatches(img1, keypoints1,  // 1st image and its keypoints
        img2, keypoints2,  // 2nd image and its keypoints
        matches,// the matches
        imageMatches2); // color of the lines
    cv::namedWindow("Filtered Matches");
    cv::imshow("Filtered Matches", imageMatches2);


    waitKey(0);

    return 0;
}
