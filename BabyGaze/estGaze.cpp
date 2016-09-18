#include <opencv2\opencv.hpp>
#include "babyPara.h"
using namespace cv;

extern double irisWidthValidThresh;
extern double irisHeightValidThresh;
extern double irisWidthDiffThresh;

int estGaze(Mat const &eyeballregion_left_bw, Mat const &eyeballregion_right_bw, Mat const &currFrame,
			 double X_eyeCenter_calL, double Y_eyeCenter_calL, 
			 double X_eyeCenter_calR, double Y_eyeCenter_calR,
			 babyPara const &currBaby) {
int EstDirection =0;	//-1=left, 1=right, 0=fail
// 1st -  determine if left/right eye is valid
	// Left Eye
	Mat labels_rowsumL, labels_colsumL;
	reduce(eyeballregion_left_bw, labels_rowsumL, 0, CV_REDUCE_SUM);//labels_rowsumL = row vector
	reduce(eyeballregion_left_bw, labels_colsumL, 1, CV_REDUCE_SUM);//labels_colsumL = col vector

	vector<int> ind_colL, ind_rowL;		// ind_colL = nonempty column index, row vector
										// ind_rowL = nonempty row index, column vector
	for (int i=0; i<labels_rowsumL.cols; i++) {
		if (labels_rowsumL.at<float>(i) >0)
			ind_colL.push_back(i);
	}
	for (int i=0; i<labels_colsumL.rows; i++) {
		if (labels_colsumL.at<float>(i) >0)
			ind_rowL.push_back(i);
	}

	double irisWidthL =0.0 , irisHeightL = 0.0;
	if (ind_rowL.size() < 2)	// iris height has to be at least 3 pixel wide
		irisHeightL = 100.0;
	else
		irisHeightL = ind_rowL.back()-ind_rowL.front();

	if (ind_colL.size() < 2)	// iris width has to be at least 3 pixel wide
		irisWidthL = 100.0;
	else
		irisWidthL = ind_colL.back()-ind_colL.front();

	bool  leftEyeValid_flag = irisWidthL<=irisWidthValidThresh && irisHeightL<=irisHeightValidThresh;

	// Right Eye
	Mat labels_rowsumR, labels_colsumR;
	reduce(eyeballregion_right_bw, labels_rowsumR, 0, CV_REDUCE_SUM);//labels_rowsumR = row vector
	reduce(eyeballregion_right_bw, labels_colsumR, 1, CV_REDUCE_SUM);//labels_colsumR = col vector
	vector<int> ind_colR, ind_rowR;
	for (int i=0; i<labels_rowsumR.cols; i++) {
		if (labels_rowsumR.at<float>(i) >0)
			ind_colR.push_back(i);
	}
	for (int i=0; i<labels_colsumR.rows; i++) {
		if (labels_colsumR.at<float>(i) >0)
			ind_rowR.push_back(i);
	}
		
	double irisWidthR =0.0 , irisHeightR = 0.0;
	if (ind_rowR.size() < 2)
		irisHeightR = 100.0;
	else
		irisHeightR = ind_rowR.back()-ind_rowR.front();

	if (ind_colR.size() < 2)
		irisWidthR = 100.0;
	else
		irisWidthR = ind_colR.back()-ind_colR.front();

	bool  rightEyeValid_flag = irisWidthR<=irisWidthValidThresh && irisHeightR<=irisHeightValidThresh;

// 2nd - find eye white region and compute the difference, then determine gaze direction
	if (!rightEyeValid_flag && !leftEyeValid_flag)
		return 0;
	else {
		double diffL=-1, diffR=-1;
		double leftInstyL=0.0, rightInstyL=0.0;
		double leftInstyR=0.0, rightInstyR=0.0;
		double tempY=0;
        double tempX=5;

		// left eye
		if (leftEyeValid_flag) {
			double leftbound_TL=ind_colL.front();
            double leftbound_T_vidL=X_eyeCenter_calL-(currBaby.half_eyeball_H_ori+1-leftbound_TL);
            double rightbound_TL=ind_colL.back();
            double rightbound_T_vidL=X_eyeCenter_calL-(currBaby.half_eyeball_H_ori+1-rightbound_TL);
            double upbound_TL=ind_rowL.front();
            double upbound_T_vidL=Y_eyeCenter_calL-(currBaby.half_eyeball_V_ori+1-upbound_TL);
            double bottombound_TL=ind_rowL.back();
            double bottombound_T_vidL=Y_eyeCenter_calL-(currBaby.half_eyeball_V_ori+1-bottombound_TL);

			Size patchSizeL(tempX,2*tempY+bottombound_T_vidL-upbound_T_vidL);
			Point2f leftPatchCenterL(leftbound_T_vidL-tempX/2.0, (upbound_T_vidL+bottombound_T_vidL)/2.0);
			Point2f rightPatchCenterL(rightbound_T_vidL+tempX/2.0, (upbound_T_vidL+bottombound_T_vidL)/2.0);
			Mat leftPatchL, rightPatchL;
			getRectSubPix(currFrame, patchSizeL, leftPatchCenterL, leftPatchL, -1);
			getRectSubPix(currFrame, patchSizeL, rightPatchCenterL, rightPatchL, -1);
			if (leftPatchL.rows*leftPatchL.cols>0 && rightPatchL.rows*rightPatchL.cols>0) {
				Mat leftPatchL_gray,rightPatchL_gray;
				cvtColor(leftPatchL, leftPatchL_gray, CV_BGR2GRAY );
				cvtColor(rightPatchL, rightPatchL_gray, CV_BGR2GRAY );
				leftPatchL_gray.convertTo(leftPatchL_gray,CV_32FC1);
				rightPatchL_gray.convertTo(rightPatchL_gray,CV_32FC1);
				leftPatchL.convertTo(leftPatchL,CV_32FC1);
				rightPatchL.convertTo(rightPatchL,CV_32FC1);

				leftPatchL=leftPatchL.reshape(0,1);
				rightPatchL=rightPatchL.reshape(0,1);
				leftPatchL_gray=leftPatchL_gray.reshape(0,1);
				rightPatchL_gray=rightPatchL_gray.reshape(0,1);
				Mat leftPatchL_gray_idx, rightPatchL_gray_idx;
				sortIdx(leftPatchL_gray, leftPatchL_gray_idx, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
				sortIdx(rightPatchL_gray, rightPatchL_gray_idx, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
				
				int counter_i=0;
				while ( counter_i<0.8*leftPatchL_gray_idx.cols) {
					double currIdx = leftPatchL_gray_idx.at<int>(counter_i);
					leftInstyL += leftPatchL.at<Vec3f>(currIdx)[0] / leftPatchL.at<Vec3f>(currIdx)[2];
					counter_i++;
				}
				leftInstyL /= counter_i-1;

				counter_i=0;
				while ( counter_i<0.8*rightPatchL_gray_idx.cols) {
					double currIdx = rightPatchL_gray_idx.at<int>(counter_i);
					rightInstyL += rightPatchL.at<Vec3f>(currIdx)[0] / rightPatchL.at<Vec3f>(currIdx)[2];
					counter_i++;
				}
				rightInstyL /= counter_i-1;

				diffL = abs(leftInstyL - rightInstyL);
			}

		}

		// right eye
		if (rightEyeValid_flag) {
			double leftbound_TR=ind_colR.front();
            double leftbound_T_vidR=X_eyeCenter_calR-(currBaby.half_eyeball_H_ori+1-leftbound_TR);
            double rightbound_TR=ind_colR.back();
            double rightbound_T_vidR=X_eyeCenter_calR-(currBaby.half_eyeball_H_ori+1-rightbound_TR);
            double upbound_TR=ind_rowR.front();
            double upbound_T_vidR=Y_eyeCenter_calR-(currBaby.half_eyeball_V_ori+1-upbound_TR);
            double bottombound_TR=ind_rowR.back();
            double bottombound_T_vidR=Y_eyeCenter_calR-(currBaby.half_eyeball_V_ori+1-bottombound_TR);
			
			Size patchSizeR(tempX,2*tempY+bottombound_T_vidR-upbound_T_vidR);
			Point2f leftPatchCenterR(leftbound_T_vidR-tempX/2.0, (upbound_T_vidR+bottombound_T_vidR)/2.0);
			Point2f rightPatchCenterR(rightbound_T_vidR+tempX/2.0, (upbound_T_vidR+bottombound_T_vidR)/2.0);
			Mat leftPatchR, rightPatchR;
			getRectSubPix(currFrame, patchSizeR, leftPatchCenterR, leftPatchR, -1);
			getRectSubPix(currFrame, patchSizeR, rightPatchCenterR, rightPatchR, -1);
			if (leftPatchR.rows*leftPatchR.cols>0 && rightPatchR.rows*rightPatchR.cols>0) {
				Mat leftPatchR_gray,rightPatchR_gray;
				cvtColor(leftPatchR, leftPatchR_gray, CV_BGR2GRAY );
				cvtColor(rightPatchR, rightPatchR_gray, CV_BGR2GRAY );
				leftPatchR_gray.convertTo(leftPatchR_gray,CV_32FC1);
				rightPatchR_gray.convertTo(rightPatchR_gray,CV_32FC1);
				leftPatchR.convertTo(leftPatchR,CV_32FC1);
				rightPatchR.convertTo(rightPatchR,CV_32FC1);

				leftPatchR=leftPatchR.reshape(0,1);
				rightPatchR=rightPatchR.reshape(0,1);
				leftPatchR_gray=leftPatchR_gray.reshape(0,1);
				rightPatchR_gray=rightPatchR_gray.reshape(0,1);
				Mat leftPatchR_gray_idx, rightPatchR_gray_idx;
				sortIdx(leftPatchR_gray, leftPatchR_gray_idx, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
				sortIdx(rightPatchR_gray, rightPatchR_gray_idx, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
			
				int counter_i=0;
				while ( counter_i<0.8*leftPatchR_gray_idx.cols) {
					double currIdx = leftPatchR_gray_idx.at<int>(counter_i);
					leftInstyR += leftPatchR.at<Vec3f>(currIdx)[0] / leftPatchR.at<Vec3f>(currIdx)[2];
					counter_i++;
				}
				leftInstyR /= counter_i-1;

				counter_i=0;
				while ( counter_i<0.8*rightPatchR_gray_idx.cols) {
					double currIdx = rightPatchR_gray_idx.at<int>(counter_i);
					rightInstyR += rightPatchR.at<Vec3f>(currIdx)[0] / rightPatchR.at<Vec3f>(currIdx)[2];
					counter_i++;
				}
				rightInstyR /= counter_i-1;

				diffR = abs(leftInstyR - rightInstyR);
			}

		}

		// Decide gaze direction
		if (leftEyeValid_flag && rightEyeValid_flag) {
				if (abs(irisWidthL-irisWidthR)>irisWidthDiffThresh) {
						if (irisWidthL<=irisWidthR) {
								if (leftInstyL>rightInstyL)
									EstDirection=1;
								else
									EstDirection=-1;
						}
						else {
								if (leftInstyR>rightInstyR)
									EstDirection=1;
								else
									EstDirection=-1; 
						}
				}
				else {
						if (diffL >= diffR) {
								if (leftInstyL>rightInstyL)
									EstDirection=1;
								else
									EstDirection=-1;
						}
						else {
								if (leftInstyR>rightInstyR)
									EstDirection=1;
								else
									EstDirection=-1;
						}
				}

		}
		else if (leftEyeValid_flag) {
				if (leftInstyL>rightInstyL)
                    EstDirection=1;
                else
                    EstDirection=-1;
		}
		else if (rightEyeValid_flag) {
				if (leftInstyR>rightInstyR)
                    EstDirection=1;
                else
                    EstDirection=-1;
		}

		return EstDirection;
	}



}