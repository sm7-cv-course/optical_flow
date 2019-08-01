#ifndef OPTICALFLOWMATCHER2D_H
#define OPTICALFLOWMATCHER2D_H

#include "sim_2d.h"
#include "opencv2/opencv.hpp"


class OpticalFlowMatcher2D : public SIM_2D
{
public:
    OpticalFlowMatcher2D();
    virtual ~OpticalFlowMatcher2D();

    virtual QVector3D estimate_pose(cv::Mat frame);
    virtual void run();
};

#endif // OPTICALFLOWMATCHER2D_H
