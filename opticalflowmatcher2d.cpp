#include "opticalflowmatcher2d.h"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv::xfeatures2d;
using namespace cv;

// On input it is eat good corners with aligned size.
bool show_img_wth_points(cv::Mat img, std::vector<cv::Point2f> const& corners0, std::vector<cv::Point2f> const& corners1) {
    cv::namedWindow("frame_gray", 1);

    // Draw tracks.
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
    for(int i=0; i < corners0.size(); ++i) {
        cv::Scalar bgrPixel(255 * rand() % 255, 255 * rand() % 255, 255 * rand() % 255);
        double dist = sqrt((corners0[i].x - corners1[i].x) * (corners0[i].x - corners1[i].x) + (corners0[i].y - corners1[i].y) * (corners0[i].y - corners1[i].y));
        cv::line(mask, corners0[i], corners1[i], bgrPixel, 2);
        cv::circle(img, corners0[i], 5, bgrPixel, -1);
    }

    cv::imshow("frame_gray", img);

    if(cv::waitKey(30) >= 0) return false;

    return true;
}

OpticalFlowMatcher2D::OpticalFlowMatcher2D()
{

}

OpticalFlowMatcher2D::~OpticalFlowMatcher2D()
{

}

QVector3D
OpticalFlowMatcher2D::estimate_pose(cv::Mat inFrame) {

    QVector3D result;

cv::VideoCapture cap(0); // open the default camera

cv::Mat frame, prev_frame, frame_gray, prev_frame_gray;
std::vector<cv::Point2f> corners0, corners1;
std::vector<unsigned char> status;
std::vector<float> err;

cap >> frame; // get a new frame from camera
cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
prev_frame = frame;
prev_frame_gray = frame_gray;

for(;;)
{
    cap >> frame; // get a new frame from camera
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    // cv::GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
    // Canny(edges, edges, 0, 30, 3);

    // goodFeaturesToTrack(prev_frame_gray, corners0, maxCorners, qualityLevel, minDistance); //, InputArray mask=noArray(), int blockSize=3, bool useHarrisDetector=false, double k=0.04 );
    // goodFeaturesToTrack(frame_gray, corners1, maxCorners, qualityLevel, minDistance); //, InputArray mask=noArray(), int blockSize=3, bool useHarrisDetector=false, double k=0.04 );

    if(! corners0.empty()) {
        cv::calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, corners0, corners1, status, err); //, Size winSize=Size(21,21), int maxLevel=3, TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), int flags=0, double minEigThreshold=1e-4);

        if(! show_img_wth_points(frame_gray, corners0, corners1)) break;
    }

    corners0 = corners1;
    //prev_frame = frame;
    prev_frame_gray = frame_gray;
    // std::swap(corners0, corners1);
    // cv::swap(prev_frame, frame);
    // cv::swap(prev_frame_gray, frame_gray);
}

    return result;
}

void
OpticalFlowMatcher2D::run() {

}
