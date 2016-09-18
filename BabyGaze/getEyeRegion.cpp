#include <opencv2\opencv.hpp>
#include "babyPara.h"
using namespace cv;

bool getEyeRegion(Mat const &Ra, Mat const &Rr, Mat const &T_curr4, Mat const &currFrame, Point2d dot_position, babyPara currBaby,// Input
				  double &X_eyeCenter_calL,		// Output
				  double &Y_eyeCenter_calL,
				  double &X_eyeCenter_calR,
				  double &Y_eyeCenter_calR,
				  Mat &eyeballregion_left_cal_3,
				  Mat &eyeballregion_right_cal_3) 
{
	const Scalar dotP (dot_position.x, dot_position.y, 0.0);
	double half_eyeball_H_ori = currBaby.half_eyeball_H_ori;
	double half_eyeball_V_ori = currBaby.half_eyeball_V_ori;
	double X_eyeCenter_offsetL = currBaby.X_eyeCenter_offsetL;
	double Y_eyeCenter_offsetL = currBaby.Y_eyeCenter_offsetL;
	double X_eyeCenter_offsetR = currBaby.X_eyeCenter_offsetR;
	double Y_eyeCenter_offsetR = currBaby.Y_eyeCenter_offsetR;
	// Left Eye
	Mat coorL_cal_all;
	coorL_cal_all = T_curr4 * Ra * Rr * currBaby.coorL_all_xyz;

	coorL_cal_all.row(0) += dot_position.x;
	coorL_cal_all.row(1) += dot_position.y;
	//add(coorL_cal_all, dotP, coorL_cal_all);			// Wrong!! 'coorL_cal_all' only have 1 channel, but add(Mat, Scalar) should be used with multiple channels.

	Mat coorL_cal_x, coorL_cal_y;
	coorL_cal_x = coorL_cal_all.rowRange(0,1).colRange(Range::all());
	coorL_cal_y = coorL_cal_all.rowRange(1,2).colRange(Range::all());
	coorL_cal_x = coorL_cal_x.reshape(0,2*half_eyeball_V_ori+1);
	coorL_cal_y = coorL_cal_y.reshape(0,2*half_eyeball_V_ori+1);

	Mat tempCL = (Mat_<float>(3,1) << X_eyeCenter_offsetL, Y_eyeCenter_offsetL, 0);
	Mat tempL = T_curr4 * Ra * Rr * tempCL;

	X_eyeCenter_calL = tempL.at<float>(0,0) + dot_position.x;
    Y_eyeCenter_calL = tempL.at<float>(1,0) + dot_position.y;
	// Check if out of currFrame
	double minX_L, maxX_L;
	double minY_L, maxY_L;
	minMaxLoc(coorL_cal_x, &minX_L);
	minMaxLoc(coorL_cal_y, &minY_L);
	minMaxLoc(coorL_cal_x, NULL, &maxX_L);
	minMaxLoc(coorL_cal_y, NULL, &maxY_L);
	
	if (minX_L<0 || minY_L<0 || maxX_L>319 || maxY_L>239) 
		return false;
	remap(currFrame, eyeballregion_left_cal_3, coorL_cal_x, coorL_cal_y, CV_INTER_LINEAR);


	// Right Eye
	Mat coorR_cal_all;
	coorR_cal_all = T_curr4 * Ra * Rr * currBaby.coorR_all_xyz;

	coorR_cal_all.row(0) += dot_position.x;
	coorR_cal_all.row(1) += dot_position.y;

	Mat coorR_cal_x, coorR_cal_y;
	coorR_cal_x = coorR_cal_all.rowRange(0,1).colRange(Range::all());
	coorR_cal_y = coorR_cal_all.rowRange(1,2).colRange(Range::all());
	coorR_cal_x = coorR_cal_x.reshape(0,2*half_eyeball_V_ori+1);
	coorR_cal_y = coorR_cal_y.reshape(0,2*half_eyeball_V_ori+1);

	Mat tempCR = (Mat_<float>(3,1) << X_eyeCenter_offsetR, Y_eyeCenter_offsetR, 0);
	Mat tempR = T_curr4 * Ra * Rr * tempCR;

    X_eyeCenter_calR = tempR.at<float>(0,0) + dot_position.x;
    Y_eyeCenter_calR = tempR.at<float>(1,0) + dot_position.y; 
	// Check if out of currFrame
	double minX_R, maxX_R;
	double minY_R, maxY_R;
	minMaxLoc(coorR_cal_x, &minX_R);
	minMaxLoc(coorR_cal_y, &minY_R);
	minMaxLoc(coorR_cal_x, NULL, &maxX_R);
	minMaxLoc(coorR_cal_y, NULL, &maxY_R);
	
	if (minX_R<0 || minY_R<0 || maxX_R>319 || maxY_R>239) 
		return false;
	remap(currFrame, eyeballregion_right_cal_3, coorR_cal_x, coorR_cal_y, CV_INTER_LINEAR);

	return true;
}