#include <opencv2\opencv.hpp>
using namespace cv;

Scalar getSkinAvg(Mat &currFrame, Point2d dot_position) {
	Mat skinRegion1, skinRegion2;
	Point2d centerL(dot_position.x -15 , dot_position.y + 15);
	Point2d centerR(dot_position.x +15 , dot_position.y + 15);
	getRectSubPix(currFrame, Size(21,11), centerL, skinRegion1, -1);
	getRectSubPix(currFrame, Size(21,11), centerR, skinRegion2, -1);

	Scalar skinAvg = (mean(skinRegion1)+mean(skinRegion2))/2.0;
	return skinAvg;
}
