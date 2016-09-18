// This file defines thresholds and some parameters for the algorithm
#ifndef __programPara_H_INCLUDED__   // if x.h hasn't been included yet...
#define __programPara_H_INCLUDED__

#include <opencv2\opencv.hpp>
#define PI 3.14159265
using namespace cv;
// Thresholds (global variable definition, with initialization)
// should make those const. Buy need to declare in a .h file, which should be included by all .cpp that use those variables.
// Then define and initialize those variables in a .cpp file. (because const implies internal linkage by default !!)
// OR: extern double const maxcorr_thresh_templateMatching=0.3, to explicitly make the const varialb global linkage. (note that 
// if not const, the extern can be omitted since those varialbes are defined outside any function, which have global linkage by default)
double maxcorr_thresh_templateMatching=0.25;
//double maxcorr_thresh_newTemplate=0.3;
double reflectionThresh=100;
double irisWidthDiffThresh=5;
double irisWidthValidThresh=20;
double irisHeightValidThresh=15;
double RegionThresh=6;
double MinIrisSize=20;

// Eyeball parameter
double z0_eye_target=0;
double R_ori[]={3,6,9,11,11+3};	// R_ori[4] & R[4] are hard coded to be 14, which means the maximum size of the target
double R[]={3,6,9,11,11+3};

// Filter Templates
static Mat get_refineDisk(double R[]) {
	Mat refineDisk(Mat::zeros(2*R[0]+1,2*R[0]+1,CV_32FC1));	// 7x7 matrix
	for (int r=0; r<refineDisk.rows; r++) {
		for (int c=0; c<refineDisk.cols; c++) {
			double currX = c - R[0];
			double currY = r - R[0];
			double currR = sqrt(currX*currX+currY*currY);
			if (currR <= R[0])
				refineDisk.at<float>(r,c) = 1;
		}
	}
	return refineDisk;
}
Mat refineDisk = get_refineDisk(R);
static Mat get_LoG (double sigma, int ksize) {
	// first calculate Gaussian
	Mat h(Mat::zeros(ksize,ksize,CV_32FC1));
	float siz = (ksize-1.0)/2.0;
	float std2 = sigma*sigma*sigma;
	float sum = 0.0;
	for (float r=-siz; r<=siz; r++) {
		for (float c=-siz; c<=siz; c++) {
			float arg = exp(-(r*r+c*c)/2.0/std2);
			h.at<float>(r+siz,c+siz) = arg;
			sum+=arg;
		}
	}
	if (sum !=0)
		h = h / sum;
	// now calculate Laplacian
	Mat h1(Mat::zeros(ksize,ksize,CV_32FC1));
	Mat LoG(Mat::zeros(ksize,ksize,CV_32FC1));
	sum = 0.0;
	for (float r=-siz; r<=siz; r++) {
		for (float c=-siz; c<=siz; c++) {
			float arg = h.at<float>(r+siz,c+siz) * (r*r+c*c-2*std2)/std2/std2;
			h1.at<float>(r+siz,c+siz) = arg;
			sum+=arg;
		}
	}
	LoG = -h1 + sum/(double)ksize/(double)ksize;
	return LoG;
}
Mat H_reflection = get_LoG (1.0, 7);

// Pre-compute slice index (for finding the cross on target)
static bool get_sliceIdx (double R[], Mat &slice_xPool, Mat &slice_yPool) {
	double origin = R[4];
	Mat slice_idx(Mat::zeros(1,1+R[3]+2,CV_32FC1));
	for (int i=0; i<=R[3]+2; i++) {
		slice_idx.at<float>(i) = (float)i;
	}

	Mat curr_xPool(Mat::zeros(360,slice_idx.cols,CV_32FC1));
	Mat curr_yPool(Mat::zeros(360,slice_idx.cols,CV_32FC1));
	int rowCnt=0;
	for (float currAngle=135; currAngle>=-224; currAngle--) {
		curr_xPool(Range(rowCnt,rowCnt+1),Range::all()) = origin + cos(currAngle/180.0*PI)*slice_idx;
		curr_yPool(Range(rowCnt,rowCnt+1),Range::all()) = origin - sin(currAngle/180.0*PI)*slice_idx;
		rowCnt++;
	}
	curr_xPool.copyTo(slice_xPool);
	curr_yPool.copyTo(slice_yPool);
	return true;
}
Mat slice_xPool, slice_yPool;
static bool dummy = get_sliceIdx(R, slice_xPool, slice_yPool);	// has to use a  return value, otherwise this line calls a funciton out 
																// of any context.
// Pre-compute slice band index (for finding the inner points on target)
static bool get_slice_xyBand(double R[], Mat &slice_xyBand) { //slice_xyBand = 2x3(R[2]+1) = 2x30
	for (int i=0; i<R[2]+1; i++) {
		slice_xyBand.at<float>(0,3*i)   = i;
		slice_xyBand.at<float>(0,3*i+1) = i;
		slice_xyBand.at<float>(0,3*i+2) = i;

		slice_xyBand.at<float>(1,3*i)   = -1;
		slice_xyBand.at<float>(1,3*i+1) = 0;
		slice_xyBand.at<float>(1,3*i+2) = 1;
	}
	return true;
}
Mat slice_xyBand(Mat::zeros(2,3*(R[2]+1),CV_32FC1));
static bool dummy2 = get_slice_xyBand(R,slice_xyBand);

#endif