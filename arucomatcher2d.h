#ifndef ARUCOMATCHER2D_H
#define ARUCOMATCHER2D_H

#include "sim_2d.h"
#include "opencv2/opencv.hpp"

class ArucoMatcher2D : public SIM_2D
{
public:
    explicit ArucoMatcher2D();
    virtual ~ArucoMatcher2D();

    // x, y, phi relative to image SC.
    virtual QVector3D estimate_pose(cv::Mat frame);
    // To be possible to implement it with QThread.
    virtual void run();

};


#endif // ARUCOMATCHER2D_H
