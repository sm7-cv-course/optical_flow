#ifndef SIM_2D_H
#define SIM_2D_H

#include <QVector3D>
#include "opencv2/opencv.hpp"


// Abstract class for 2d pose estimator.
class SIM_2D
{
public:
    SIM_2D(){;}
    virtual ~SIM_2D(){;}

public:
    // x, y, phi relative to image SC.
    virtual QVector3D estimate_pose(cv::Mat frame)=0;
    // To be possible to implement it with QThread.
    virtual void run()=0;
};

#endif // SIM_2D_H
