#include <opencv2\opencv.hpp>
#include <math.h>
using namespace cv;
using namespace std;
extern double reflectionThresh;
extern Mat H_reflection;
string type2str(int type);

void removeReflection(Mat &eyeballregion_left_cal_3, Mat &eyeballregion_right_cal_3) {
	Mat tempeye1L, tempeye1R;
	cvtColor(eyeballregion_left_cal_3, tempeye1L, CV_BGR2GRAY);
	cvtColor(eyeballregion_right_cal_3, tempeye1R, CV_BGR2GRAY);
	Mat tempeye2L, tempeye2R;
	equalizeHist(tempeye1L, tempeye2L);
	equalizeHist(tempeye1R, tempeye2R);

	Mat mapL, mapR;
	double scale = 1.0 / pow(2.0, 10.0);
	filter2D(tempeye2L, mapL, CV_32FC1, H_reflection);
	filter2D(tempeye2R, mapR, CV_32FC1, H_reflection);

	vector<Point2f> idxL, idxR;
	for (int r=0; r<mapL.rows; r++) {
		for (int c=0; c<mapL.cols; c++) {
			if (mapL.at<float>(r,c) >= reflectionThresh)
				idxL.push_back(Point2f(c,r));		// Points are (x,y) = (col, row) 
		}
	}
	for (int r=0; r<mapR.rows; r++) {
		for (int c=0; c<mapR.cols; c++) {
			if (mapR.at<float>(r,c) >= reflectionThresh)
				idxR.push_back(Point2f(c,r));		// Points are (x,y) = (col, row) 
		}
	}

	Mat refLocStd_L, meanL;
	Mat refLocStd_R, meanR;
	meanStdDev(idxL, meanL, refLocStd_L);
	meanStdDev(idxR, meanR, refLocStd_R);

	Mat idxL_mat(idxL);
	Mat idxR_mat(idxR);
	idxL_mat=idxL_mat.reshape(1,idxL_mat.rows);
	idxR_mat=idxR_mat.reshape(1,idxR_mat.rows);

	Mat reflectionL_center, reflectionR_center;
	if (idxL.size() >= 1) {
		Mat labelsL;
		if (refLocStd_L.at<double>(0) < 1 &&  refLocStd_L.at<double>(1) < 1) {
			reflectionL_center = Mat(Point2f(meanL.at<double>(0),meanL.at<double>(1)));
			reflectionL_center=reflectionL_center.reshape(1,reflectionL_center.rows).t();
		}	
		else {
			double kms_score = kmeans(idxL_mat, 2, labelsL, TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),	// currently only find 2 reflec points
									  1, KMEANS_PP_CENTERS, reflectionL_center);
		}
	}
	
	if (idxR.size() >= 1) {
		Mat labelsR;
		if (refLocStd_R.at<double>(0) < 1 &&  refLocStd_R.at<double>(1) < 1) {
			reflectionR_center = Mat(Point2f(meanR.at<double>(0),meanR.at<double>(1)));
			reflectionR_center=reflectionR_center.reshape(1,reflectionR_center.rows).t();
		}	
		else {
			double kms_score = kmeans(idxR_mat, 2, labelsR, TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
									  1, KMEANS_PP_CENTERS, reflectionR_center);
		}
	}
	
	for (int i=0; i<reflectionL_center.rows; i++) {
		float row_center = reflectionL_center.at<float>(i,1);
		float col_center = reflectionL_center.at<float>(i,0);
		if (row_center>0 && row_center<eyeballregion_left_cal_3.rows-1 && col_center>0 && col_center<eyeballregion_left_cal_3.cols-1)
			eyeballregion_left_cal_3.rowRange(row_center-1,row_center+2).colRange(col_center-1,col_center+2) = 0.0;
	}

	for (int i=0; i<reflectionR_center.rows; i++) {
		float row_center = reflectionR_center.at<float>(i,1);
		float col_center = reflectionR_center.at<float>(i,0);
		if (row_center>0 && row_center<eyeballregion_right_cal_3.rows-1 && col_center>0 && col_center<eyeballregion_right_cal_3.cols-1)
			eyeballregion_right_cal_3.rowRange(row_center-1,row_center+2).colRange(col_center-1,col_center+2) = 0.0;
	}
	
}
/*

cout << "meanL = " << endl << " " << meanL << endl << endl;
	cout << "refLocStd_L = " << endl << " " << refLocStd_L << endl << endl;

	cout << "meanR = " << endl << " " << meanR << endl << endl;
	cout << "refLocStd_R = " << endl << " " << refLocStd_R << endl << endl;

	string currFrame_gray_type =  type2str( meanL.type() );
	printf("meanL: %s %dx%d \n", currFrame_gray_type.c_str(), meanL.rows, meanL.cols );

	*/
