#include <opencv2\opencv.hpp>
#include "babyPara.h"
#include "ellipsePara.h"

using namespace cv;
#define PI 3.14159265

Mat genTarTemplate(double R[], double alpha, double gama);
bool tracking(Point2d &dot_position, double &corr_out, double startX_bk, double startY_bk, Mat pf, Mat nf, Mat &templateImg, double R[], Mat const &refineDisk);
void myEllipseAngleEst(ellipsePara &ep, Mat const &currTarget, double R[]);
void find8Points(Mat const &currTarget, double R[], vector<Point2f> &xyo_curr, vector<Point2f> &xy_curr);	//cv::Vector doesn't work. Must use std::vector


void on_mouse( int event, int x, int y, int flags, void *ptr ) {
	if (event == CV_EVENT_LBUTTONDOWN) {
		Point2f *p = (Point2f*)ptr;
		p->x = x;
		p->y = y;
	}
}

// return value: -1 = video read/parameter set fail, 1 = parameter successfully set
int SetBabyParameter(bool recalibrate_flag, int const babyIDnum, babyPara &currBaby, double R[], Mat refineDisk) {
	if (!recalibrate_flag){
		currBaby.populatePara(babyIDnum);
		return 1;
	}
	// initialization
	Mat TarTemplate = genTarTemplate(R,0,0);
	currBaby.populatePara(babyIDnum);
	VideoCapture currVideo("F:/gkx/UIUC study/2016 spring/Ikids/Learning Data Collection/Target/"+to_string(babyIDnum)+"/"+currBaby.videoNames[0]);	// assuming 1st video contains frontal face
	Mat currFrame, preFrame;
	bool success = currVideo.read(preFrame);
	if (!success)
		return -1;
	cvtColor(preFrame, preFrame, CV_BGR2GRAY );
	preFrame.convertTo(preFrame,CV_32FC1);
	Point2d dot_position_pre(160, 120), dot_position(0.0,0.0);
	// Specify position relationship between target and eye
	while (1) {
		bool success = currVideo.read(currFrame);
		if (!success)
			return -1;
		int currFrameNumber = currVideo.get(CV_CAP_PROP_POS_FRAMES);

		Mat currFrame_gray;
		cvtColor(currFrame, currFrame_gray, CV_BGR2GRAY );
		currFrame_gray.convertTo(currFrame_gray,CV_32FC1);
		// tracking
		double corr;
		success = tracking(dot_position, corr, dot_position_pre.x, dot_position_pre.y, preFrame, currFrame_gray, TarTemplate, R, refineDisk);
		if (!success) {			// tracking & template matching both fail
			cout<<"tracking fail"<<endl;
			preFrame = currFrame_gray;
			dot_position_pre=dot_position;
			continue;
		}
		// ellipse fitting
		Mat currTarget;
		ellipsePara ep_curr;
		getRectSubPix(currFrame_gray, Size(2.0*R[4]+1.0,2.0*R[4]+1.0), dot_position, currTarget, -1);
		myEllipseAngleEst(ep_curr, currTarget, R);
		// if the target appears to be a circle, let user set baby parameters. Otherwise next frame
		double alpha_degree = abs( ep_curr.alpha_est / PI * 180.0);
		double gama_degree = abs( ep_curr.gama_est / PI * 180.0);


		if (alpha_degree < 10 && gama_degree<5) {
			// user select Target/Left eye/Right eye centers
			Point2f leftEye_center(-1,-1), rightEye_center(-1,-1), target_center(-1,-1);
			namedWindow("currFrame");

			setMouseCallback("currFrame",on_mouse, &target_center);
			cout<<" click Target center"<<endl;
			while (target_center.x ==-1) {
				imshow("currFrame",currFrame);
				waitKey(1);
			}

			setMouseCallback("currFrame",on_mouse, &leftEye_center );
			cout<<" click baby's Left eye center"<<endl;
			while (leftEye_center.x ==-1) {
				imshow("currFrame",currFrame);
				waitKey(1);
			}

			setMouseCallback("currFrame",on_mouse, &rightEye_center );
			cout<<" click baby's Right eye center"<<endl;
			while (rightEye_center.x ==-1) {
				imshow("currFrame",currFrame);
				waitKey(1);
			}
			// algorithm find 8 feature points on currTarget
			std::vector<Point2f> xyo_curr(4), xy_curr(4);
			find8Points(currTarget,R,xyo_curr,xy_curr);
			// show all the centers, if accepted, return 1. Otherwise try next frame
			circle(currFrame, leftEye_center, 1, Scalar(0,255,0), 1);
			circle(currFrame, rightEye_center, 1, Scalar(0,255,0), 1);
			circle(currFrame, target_center, 1, Scalar(0,255,0), 1);

			Point2f p1=target_center+xy_curr[0];
			Point2f p2=target_center+xy_curr[1];
			Point2f p3=target_center+xy_curr[2];
			Point2f p4=target_center+xy_curr[3];
			circle(currFrame, p1, 1, Scalar(0,255,0), 1);
			circle(currFrame, p2, 1, Scalar(0,255,0), 1);
			circle(currFrame, p3, 1, Scalar(0,255,0), 1);
			circle(currFrame, p4, 1, Scalar(0,255,0), 1);
			namedWindow("currFrame");
			imshow("currFrame",currFrame);
			waitKey(1);
			cout<<" Accept feature points (y/n) ?"<<endl;
			string userDecision;
			getline(cin, userDecision);
			while (  userDecision != "y" && userDecision != "n") {
				cout<< "Please enter 'y' or 'n' "<<endl;
				getline(cin, userDecision);
			}
			if ( userDecision=="n") {
				continue;
			}
			else {
				currBaby.dot_position = target_center;
				currBaby.centerX_left = leftEye_center.x;
				currBaby.centerY_left = leftEye_center.y;
				currBaby.centerX_right = rightEye_center.x;
				currBaby.centerY_right = rightEye_center.y;

				currBaby.xy_init = xy_curr;
				currBaby.xyo_init = xyo_curr;

				currBaby.reComputePara();

				return 1;
			}

		}

	}






	
}