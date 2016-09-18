
#include <opencv2\opencv.hpp>
#include <iostream>
#define PI 3.14159265
using namespace cv;
using namespace std;
extern Mat slice_xPool, slice_yPool, slice_xyBand;	//pre-computed x,y coordinage 

void find8Points(Mat const &currTarget, double R[], vector<Point2f> &xyo_curr, vector<Point2f> &xy_curr) {
	int num_angle = slice_xPool.rows;
	int num_sample = slice_xPool.cols;
	Mat radial_sampleAll(Mat::zeros(slice_xPool.size(),CV_32FC1));
	Mat scoreAngle(Mat::zeros(1,num_angle,CV_32FC1));
// 1- find the cross on target
	for (int i=0; i<num_angle; i++) {
		Mat radial_sample, aux;
		remap(currTarget, radial_sample, slice_xPool(Range(i,i+1),Range::all())
									   , slice_yPool(Range(i,i+1),Range::all()), CV_INTER_LINEAR);
		aux = radial_sampleAll.rowRange(i,i+1).colRange(Range::all());

		radial_sample.copyTo(aux);

		Mat mean, std;		// 'std' not consistent with Matlab ??????????????????????????????????????
		meanStdDev(radial_sample(Range::all(),Range(0,R[2]+1)), mean, std);
		float temp = mean.at<double>(0) * std.at<double>(0);
		scoreAngle.at<float>(i) = temp;
	}
	Mat log_scoreAngle;
	log(scoreAngle,log_scoreAngle);
	scoreAngle =25-log_scoreAngle;
	Mat mean_score, std_score;
	meanStdDev(scoreAngle, mean_score, std_score);
	
	Point currAngle1_idx, currAngle2_idx, currAngle3_idx, currAngle4_idx;	//'currAngle3_idx.x' = row # in 'slice_xPool/slice_yPool' corresponding to the cross angle
	minMaxLoc(scoreAngle(Range::all(),Range(0,90)), NULL, NULL, NULL, &currAngle1_idx);
	minMaxLoc(scoreAngle(Range::all(),Range(90,180)), NULL, NULL, NULL, &currAngle2_idx);
		currAngle2_idx.x += 90;
	minMaxLoc(scoreAngle(Range::all(),Range(180,270)), NULL, NULL, NULL, &currAngle3_idx);
		currAngle3_idx.x += 180;
	minMaxLoc(scoreAngle(Range::all(),Range(270,scoreAngle.cols)), NULL, NULL, NULL, &currAngle4_idx);
		currAngle4_idx.x += 270;
// 2- find 4 outter points
	Mat radial_sample1 = radial_sampleAll(Range(currAngle1_idx.x,currAngle1_idx.x+1),Range::all());
	Mat radial_sample2 = radial_sampleAll(Range(currAngle2_idx.x,currAngle2_idx.x+1),Range::all());
	Mat radial_sample3 = radial_sampleAll(Range(currAngle3_idx.x,currAngle3_idx.x+1),Range::all());
	Mat radial_sample4 = radial_sampleAll(Range(currAngle4_idx.x,currAngle4_idx.x+1),Range::all());
	Point p1oIdx, p2oIdx, p3oIdx, p4oIdx;	// index of outter point along each radial sample
	minMaxLoc(radial_sample1, NULL, NULL, NULL, &p1oIdx);
	minMaxLoc(radial_sample2, NULL, NULL, NULL, &p2oIdx);
	minMaxLoc(radial_sample3, NULL, NULL, NULL, &p3oIdx);
	minMaxLoc(radial_sample4, NULL, NULL, NULL, &p4oIdx);
	
	// for output, -14 to 14
	xyo_curr[0].x = slice_xPool.at<float>(currAngle1_idx.x,p1oIdx.x) - R[4];
	xyo_curr[0].y = slice_yPool.at<float>(currAngle1_idx.x,p1oIdx.x) - R[4];
	xyo_curr[1].x = slice_xPool.at<float>(currAngle2_idx.x,p1oIdx.x) - R[4];
	xyo_curr[1].y = slice_yPool.at<float>(currAngle2_idx.x,p1oIdx.x) - R[4];
	xyo_curr[2].x = slice_xPool.at<float>(currAngle3_idx.x,p1oIdx.x) - R[4];
	xyo_curr[2].y = slice_yPool.at<float>(currAngle3_idx.x,p1oIdx.x) - R[4];
	xyo_curr[3].x = slice_xPool.at<float>(currAngle4_idx.x,p1oIdx.x) - R[4];
	xyo_curr[3].y = slice_yPool.at<float>(currAngle4_idx.x,p1oIdx.x) - R[4];
	
	/*
	// for debug. w.r.t currTarget, 0-29
	xyo_curr[0].x = slice_xPool.at<float>(currAngle1_idx.x,p1oIdx.x);
	xyo_curr[0].y = slice_yPool.at<float>(currAngle1_idx.x,p1oIdx.x);
	xyo_curr[1].x = slice_xPool.at<float>(currAngle2_idx.x,p1oIdx.x);
	xyo_curr[1].y = slice_yPool.at<float>(currAngle2_idx.x,p1oIdx.x);
	xyo_curr[2].x = slice_xPool.at<float>(currAngle3_idx.x,p1oIdx.x);
	xyo_curr[2].y = slice_yPool.at<float>(currAngle3_idx.x,p1oIdx.x);
	xyo_curr[3].x = slice_xPool.at<float>(currAngle4_idx.x,p1oIdx.x);
	xyo_curr[3].y = slice_yPool.at<float>(currAngle4_idx.x,p1oIdx.x);
	*/
// 3- find 4 inner points
	float currAngle[4];		// Angles are [135:-1:-224] hard coded
	currAngle[0] = (135.0 - currAngle1_idx.x)/180.0*PI;
	currAngle[1] = (135.0 - currAngle2_idx.x)/180.0*PI;
	currAngle[2] = (135.0 - currAngle3_idx.x)/180.0*PI;
	currAngle[3] = (135.0 - currAngle4_idx.x)/180.0*PI;

	Mat RH2 = (Mat_<float>(2,2) << cos(currAngle[1]), sin(currAngle[1]), -sin(currAngle[1]), cos(currAngle[1]));
	Mat RH4 = (Mat_<float>(2,2) << cos(currAngle[3]), sin(currAngle[3]), -sin(currAngle[3]), cos(currAngle[3]));
	Mat RV1 = (Mat_<float>(2,2) << cos(currAngle[0]), sin(currAngle[0]), -sin(currAngle[0]), cos(currAngle[0]));
	Mat RV3 = (Mat_<float>(2,2) << cos(currAngle[2]), sin(currAngle[2]), -sin(currAngle[2]), cos(currAngle[2]));

	Mat curr_H2=R[4]+RH2*slice_xyBand;
    Mat curr_H4=R[4]+RH4*slice_xyBand;
    Mat curr_V1=R[4]+RV1*slice_xyBand;
    Mat curr_V3=R[4]+RV3*slice_xyBand;

	Mat radial_sampleH2, radial_sampleH4, radial_sampleH1, radial_sampleH3;
	remap(currTarget, radial_sampleH2, curr_H2(Range(0,1),Range::all())
									   , curr_H2(Range(1,2),Range::all()), CV_INTER_LINEAR);
	remap(currTarget, radial_sampleH4, curr_H4(Range(0,1),Range::all())
									   , curr_H4(Range(1,2),Range::all()), CV_INTER_LINEAR);
	remap(currTarget, radial_sampleH1, curr_V1(Range(0,1),Range::all())
									   , curr_V1(Range(1,2),Range::all()), CV_INTER_LINEAR);
	remap(currTarget, radial_sampleH3, curr_V3(Range(0,1),Range::all())
									   , curr_V3(Range(1,2),Range::all()), CV_INTER_LINEAR);
	int Nrow = radial_sampleH2.total()/3;
	radial_sampleH2 = radial_sampleH2.reshape(0,Nrow).t();
	radial_sampleH4 = radial_sampleH4.reshape(0,Nrow).t();
	radial_sampleH1 = radial_sampleH1.reshape(0,Nrow).t();
	radial_sampleH3 = radial_sampleH3.reshape(0,Nrow).t();

	Mat radialH2score=radial_sampleH2.row(0)+radial_sampleH2.row(2)-radial_sampleH2.row(1);
	Mat radialH4score=radial_sampleH4.row(0)+radial_sampleH4.row(2)-radial_sampleH4.row(1);
	Mat radialH1score=radial_sampleH1.row(0)+radial_sampleH1.row(2)-radial_sampleH1.row(1);
	Mat radialH3score=radial_sampleH3.row(0)+radial_sampleH3.row(2)-radial_sampleH3.row(1);
    
	Sobel(radialH2score, radialH2score, CV_32FC1, 1, 0, 1);
	Sobel(radialH4score, radialH4score, CV_32FC1, 1, 0, 1);
	Sobel(radialH1score, radialH1score, CV_32FC1, 1, 0, 1);
	Sobel(radialH3score, radialH3score, CV_32FC1, 1, 0, 1);
	
	int deadZone = R[0]-1;

	Point p4ind, p2ind, p3ind, p1ind;
	minMaxLoc(radialH4score(Range::all(),Range(deadZone,radialH4score.cols)), NULL, NULL, &p4ind, NULL);
		p4ind.x += deadZone;
	minMaxLoc(radialH2score(Range::all(),Range(deadZone,radialH2score.cols)), NULL, NULL, &p2ind, NULL);
		p2ind.x += deadZone;
	minMaxLoc(radialH3score(Range::all(),Range(deadZone,radialH3score.cols)), NULL, NULL, &p3ind, NULL);
		p3ind.x += deadZone;
	minMaxLoc(radialH1score(Range::all(),Range(deadZone,radialH1score.cols)), NULL, NULL, &p1ind, NULL);
		p1ind.x += deadZone;

	// for output, -14 to 14
	xy_curr[0].x = cos(currAngle[0])*p1ind.x;
	xy_curr[0].y = - sin(currAngle[0])*p1ind.x;
	xy_curr[1].x = cos(currAngle[1])*p2ind.x;
	xy_curr[1].y = - sin(currAngle[1])*p2ind.x;
	xy_curr[2].x = cos(currAngle[2])*p3ind.x;
	xy_curr[2].y = - sin(currAngle[2])*p3ind.x;
	xy_curr[3].x = cos(currAngle[3])*p4ind.x;
	xy_curr[3].y = - sin(currAngle[3])*p4ind.x;
	
	/*
	// for debug, w.r.t currTarget, 0 to 29
	xy_curr[0].x = R[4] + cos(currAngle[0])*p1ind.x;
	xy_curr[0].y = R[4] - sin(currAngle[0])*p1ind.x;
	xy_curr[1].x = R[4] + cos(currAngle[1])*p2ind.x;
	xy_curr[1].y = R[4] - sin(currAngle[1])*p2ind.x;
	xy_curr[2].x = R[4] + cos(currAngle[2])*p3ind.x;
	xy_curr[2].y = R[4] - sin(currAngle[2])*p3ind.x;
	xy_curr[3].x = R[4] + cos(currAngle[3])*p4ind.x;
	xy_curr[3].y = R[4] - sin(currAngle[3])*p4ind.x;
	*/
}

/*
FILE * (file);
	file=fopen("radial_sample.txt","w");
	for (int r=0;r<radial_sample.rows;r++) {
		for (int c=0;c<radial_sample.cols;c++) {
			double currval=radial_sample.at<float>(r,c);
			fprintf (file, "%f, ",currval);
		}
		fprintf (file,"\n");
	}
fclose (file);
cout << "mean = " << endl << " " << mean << endl << endl;
cout << "std = " << endl << " " << std << endl << endl;
cout << "temp = " << endl << " " << temp << endl << endl;
*/