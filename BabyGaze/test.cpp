/*
#include <locale.h>
#include <config4cpp/Configuration.h>
#include <iostream>
#include <string> 
#include <vector>
#include <opencv2\opencv.hpp>
#include <complex>
#include "ellipsePara.h"
#include "programPara.h"
#include "babyPara.h"
using namespace config4cpp;
using namespace std;
using namespace cv;
#define PI 3.14159265

// function forward declaration
Mat genTarTemplate(double R[], double alpha, double gama);
bool tracking(Point2d &dot_position, double &corr_out, double startX_bk, double startY_bk, Mat pf, Mat nf, Mat &templateImg, double R[], Mat const &refineDisk);
void myEllipseAngleEst(ellipsePara &ep, Mat currTarget, double R[]);
void find8Points(Mat const &currTarget, double R[], vector<Point2f> &xyo_curr, vector<Point2f> &xy_curr);	//cv::Vector doesn't work. Must use std::vector
void estInPlaneRot(Mat const &Ra, Mat const &Rr, vector<Point2f> const &xyo_curr, 
									 vector<Point2f> const &xy_curr,
									 vector<Point2f> const &xy_init, 
									 vector<Point2f> const &xyo_init, 
									 Mat &T_curr4);
bool getEyeRegion(Mat const &Ra, Mat const &Rr, Mat const &T_curr4, Mat const &currFrame, Point2d dot_position, babyPara currBaby,	// Input
				  double &X_eyeCenter_calL,		// Output
				  double &Y_eyeCenter_calL,
				  double &X_eyeCenter_calR,
				  double &Y_eyeCenter_calR,
				  Mat &eyeballregion_left_cal_3,
				  Mat &eyeballregion_right_cal_3);
Scalar getSkinAvg(Mat &currFrame, Point2d dot_position);
void irisSegmentation(Mat &eyeballregion_left_cal_3, Mat &eyeballregion_right_cal_3, Scalar skinAvg,		// Input
					  Mat &eyeballregion_left_bw, Mat &eyeballregion_right_bw);							// Output
int estGaze(Mat const &eyeballregion_left_bw, Mat const &eyeballregion_right_bw, Mat const &currFrame,
			 double X_eyeCenter_calL, double Y_eyeCenter_calL, 
			 double X_eyeCenter_calR, double Y_eyeCenter_calR,
			 babyPara const &currBaby);
// for debug only
string type2str(int type);
//
void main() {
	Mat TarTemplate = genTarTemplate(R,0,0);

	int babyIDnum=35;
	babyPara currBaby;
	currBaby.populatePara(babyIDnum);
	
	for (int video_i=0; video_i<currBaby.videoNames.size(); video_i++) {
		VideoCapture currVideo("F:/gkx/UIUC study/2016 spring/Ikids/Learning Data Collection/Target/"+to_string(babyIDnum)+"/"+currBaby.videoNames[video_i]);
		int totalFrame = currVideo.get(CV_CAP_PROP_FRAME_COUNT);
		cout<<"totalFrame = "<<totalFrame<<endl;
		
		Mat currFrame, preFrame;
		// initialization
		bool success = currVideo.read(preFrame);
		if (!success)
			return;	
		cvtColor(preFrame, preFrame, CV_BGR2GRAY );
		////preFrame.convertTo(preFrame,CV_64FC1);
		preFrame.convertTo(preFrame,CV_32FC1);
		Point2d dot_position_pre(160, 120), dot_position(0.0,0.0);
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		while (1) {
			bool success = currVideo.read(currFrame);		//  currFrame= CV_8UC3
			if (!success)
				break;
			int currFrameNumber = currVideo.get(CV_CAP_PROP_POS_FRAMES);
			cout<<"currFrameNumber = "<<currFrameNumber<<endl;

			Mat currFrame_gray;
			cvtColor(currFrame, currFrame_gray, CV_BGR2GRAY );
			////currFrame_gray.convertTo(currFrame_gray,CV_64FC1);
			currFrame_gray.convertTo(currFrame_gray,CV_32FC1);
			// tracking
			double corr;																					//pos_corr={newX,newY,corr};
			success = tracking(dot_position, corr, dot_position_pre.x, dot_position_pre.y, preFrame, currFrame_gray, TarTemplate, R, refineDisk);
			if (!success) {			// tracking & template matching both fail
				cout<<"fail"<<endl;
				namedWindow("currFrame",CV_WINDOW_AUTOSIZE);
				imshow("currFrame",currFrame);
				waitKey(1);

				preFrame = currFrame_gray;
				dot_position_pre=dot_position;
				continue;
			}
			// ellipse fitting
			Mat currTarget, newTarget;
			ellipsePara ep_curr;
			getRectSubPix(currFrame_gray, Size(2.0*R[4]+1.0,2.0*R[4]+1.0), dot_position, currTarget, -1);
			myEllipseAngleEst(ep_curr, currTarget, R);
			TarTemplate = genTarTemplate(R,ep_curr.alpha_est,ep_curr.gama_est);
			// find feature points on currTarget
			std::vector<Point2f> xyo_curr(4), xy_curr(4);
			find8Points(currTarget,R,xyo_curr,xy_curr);
			// estimate in-plane rotation
			Mat Ra = (Mat_<float>(3,3) << 1.0,					  0.0,					  0.0,
										  0.0,					  cos(ep_curr.alpha_est), sin(ep_curr.alpha_est),
										  0.0,					 -sin(ep_curr.alpha_est), cos(ep_curr.alpha_est));

			Mat Rr = (Mat_<float>(3,3) << cos(ep_curr.gama_est),  0.0,					  -sin(ep_curr.gama_est),
										  0.0,					  1.0,					   0.0,
										  sin(ep_curr.gama_est),  0.0,					   cos(ep_curr.gama_est));
			Mat T_curr4;
			estInPlaneRot(Ra, Rr, xyo_curr, xy_curr, currBaby.xy_init, currBaby.xyo_init, T_curr4);
			// get eye region
			double X_eyeCenter_calL, Y_eyeCenter_calL, X_eyeCenter_calR, Y_eyeCenter_calR;
			Mat eyeballregion_left_cal_3, eyeballregion_right_cal_3;
			bool getEyeSuccess = getEyeRegion(Ra, Rr, T_curr4, currFrame, dot_position, currBaby,// Input
											  X_eyeCenter_calL,		// Output
											  Y_eyeCenter_calL,
											  X_eyeCenter_calR,
											  Y_eyeCenter_calR,
											  eyeballregion_left_cal_3,
											  eyeballregion_right_cal_3);
			if (!getEyeSuccess) {
				cout<<"eye crop failure"<<endl;
				continue;
			}
			// iris segmentation
			Scalar skinAvg = getSkinAvg(currFrame, dot_position);

			Mat eyeballregion_left_bw, eyeballregion_right_bw;
			irisSegmentation(eyeballregion_left_cal_3, eyeballregion_right_cal_3, skinAvg,		// Input
							 eyeballregion_left_bw, eyeballregion_right_bw);					// Output
			// estimate gaze direction
			
			int estSuccess = estGaze(eyeballregion_left_bw, eyeballregion_right_bw, currFrame,
									 X_eyeCenter_calL, Y_eyeCenter_calL, 
									 X_eyeCenter_calR, Y_eyeCenter_calR,
									 currBaby);
			//
			if (estSuccess==-1){
				currFrame.rowRange(Range::all()).colRange(Range(0,20))=255;
			}
			else if (estSuccess==1) {
				currFrame.rowRange(Range::all()).colRange(Range(300,320))=255;
			}


			Point2f centerL (X_eyeCenter_calL, Y_eyeCenter_calL);
			Point2f centerR (X_eyeCenter_calR, Y_eyeCenter_calR);

			circle(currFrame, dot_position, 2, Scalar(255,0,0), 2);
			circle(currFrame, centerL, 1, Scalar(0,255,0), 2);
			circle(currFrame, centerR, 1, Scalar(0,255,0), 2);
			currTarget.convertTo(currTarget,CV_8UC1);
		
			Point2f temp = dot_position;
			Point2f p1=temp+xy_curr[0];
			Point2f p2=temp+xy_curr[1];
			Point2f p3=temp+xy_curr[2];
			Point2f p4=temp+xy_curr[3];

			circle(currFrame, p1, 1, Scalar(0,255,0), 2);
			circle(currFrame, p2, 1, Scalar(0,255,0), 2);
			circle(currFrame, p3, 1, Scalar(0,255,0), 2);
			circle(currFrame, p4, 1, Scalar(0,255,0), 2);

		
			namedWindow("currFrame",CV_WINDOW_AUTOSIZE);
			//namedWindow("TarTemplate",CV_WINDOW_AUTOSIZE);
			namedWindow("left eye",CV_WINDOW_AUTOSIZE);
			namedWindow("right eye",CV_WINDOW_AUTOSIZE);
			namedWindow("eyeballregion_left_bw",CV_WINDOW_AUTOSIZE);
			namedWindow("eyeballregion_right_bw",CV_WINDOW_AUTOSIZE);
			imshow("currFrame",currFrame);
			//imshow("TarTemplate",TarTemplate);
			eyeballregion_left_cal_3.convertTo(eyeballregion_left_cal_3,CV_8UC3);
			eyeballregion_right_cal_3.convertTo(eyeballregion_right_cal_3,CV_8UC3);
			imshow("left eye",eyeballregion_left_cal_3);
			imshow("right eye",eyeballregion_right_cal_3);
			imshow("eyeballregion_left_bw",255*eyeballregion_left_bw);
			imshow("eyeballregion_right_bw",255*eyeballregion_right_bw);
		
			waitKey(0);
			
		}
	}

	

}
	//namedWindow("TarTemplate",CV_WINDOW_AUTOSIZE);
	//imshow("TarTemplate",TarTemplate);

	//string currFrame_gray_type =  type2str( currFrame_gray.type() );
	//printf("Matrix: %s %dx%d \n", currFrame_gray_type.c_str(), currFrame_gray.rows, currFrame_gray.cols );

/*
void main() {
	double R[]={3,6,9,11,11+3};
	double alpha = 5.0/180.0*PI;
	double gama = 20.0/180.0*PI;
	Mat currTarget = genTarTemplate(R,alpha,gama);

FILE * (file);
	file=fopen("currTarget.txt","w");
	for (int r=0;r<slice_xPool.rows;r++) {
		for (int c=0;c<slice_xPool.cols;c++) {
			double currval=slice_xPool.at<float>(r,c);
			fprintf (file, "%1.3f, ",currval);
		}
		fprintf (file,"\n");
	}
fclose (file);

	ellipsePara ep1;
	myEllipseAngleEst(ep1, currTarget, R);
	double a_est = ep1.alpha_est/PI*180.0;
	double r_est = ep1.gama_est/PI*180.0;
	cout<<"a_est = "<<a_est<<endl;
	cout<<"r_est = "<<r_est<<endl;
	system("pause");
}
*/
/*
void main() {
	// read values from .cfg file
	int centerX_left=-1;
	int babyIDnum=28;
	string babyIDtemp = "baby_" + to_string(babyIDnum);
	const char*  babyID= babyIDtemp.c_str();

	 Configuration * cfg = Configuration::create();
	 try{	
		cfg->parse("C:\\Users\\kevin\\Documents\\Visual Studio 2012\\Projects\\BabyGaze\\BabyGaze\\babyProfile.cfg");
		centerX_left = cfg->lookupInt(babyID, "centerX_left");
	 } catch(const ConfigurationException & ex) {
		cout << ex.c_str() << endl;
	 }
	vector<cv::Point2f>  xy_init;
	vector<cv::Point2f>  xyo_init;
	StringVector x_cross, y_cross, x_outter, y_outter;
	cfg->lookupList(babyID, "x_cross", x_cross);
	cfg->lookupList(babyID, "y_cross", y_cross);
	cfg->lookupList(babyID, "x_outter", x_outter);
	cfg->lookupList(babyID, "y_outter", y_outter);
	for (int i=0; i<4; i++) {
		cv::Point2f xy_temp(atof(x_cross[i]), atof(y_cross[i]));
		cv::Point2f xyo_temp(atof(x_outter[i]), atof(y_outter[i]));

		xy_init.push_back(xy_temp);
		xyo_init.push_back(xyo_temp);
	}
	cout<<xy_init<<endl;
	cfg->destroy();
	
	system("pause");
}
*/
/*
void main() {
	vector<Point2f> points1(3),points2(2);
	points1[0].x=1;points1[0].y=2;
	points1[1].x=3;points1[1].y=4;
	points1[2].x=5;points1[2].y=6;

	points2[0].x=7;points2[0].y=8;
	points2[1].x=9;points2[1].y=10;

	Mat temp1 = Mat(points1).reshape(1);
	Mat temp2 = Mat(points2).reshape(1);
	Mat temp3(Mat::zeros(5,3,CV_32FC1));
	vconcat(temp1,temp2,temp3.colRange(0,2).rowRange(Range::all()));

	cout << "temp1 = " << endl << " " << temp1 << endl << endl;
	cout << "temp3 = " << endl << " " << temp3 << endl << endl;

	string currFrame_gray_type =  type2str( temp1.type() );
	printf("Matrix: %s %dx%d \n", currFrame_gray_type.c_str(), temp1.rows, temp1.cols );
	

	 Mat A = (Mat_<double>(3,2) << 1,2,3,4,5,6);
	 Mat B = (Mat_<double>(3,2) << 0,0,0,0,0,0);
	 A=A.reshape(0,1);
	 //A(Range::all(),Range(0,1)) = B(Range::all(),Range(1,2));	//doesn't work!!
	 //Mat aux = A.rowRange(Range::all()).colRange(1,2);
	 //B(Range::all(),Range(0,1)).copyTo(aux);
	
     cout << "A = " << endl << " " << A << endl << endl;
	 A=A.reshape(0,3);
     cout << "A = " << endl << " " << A << endl << endl;
	 //Mat x;
	 //solve(A,B,x);
     //cout << "x = " << endl << " " << x << endl << endl;
	 system("pause");
}
*/
