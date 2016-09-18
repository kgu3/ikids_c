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

#include <winsock2.h>
#pragma comment (lib, "ws2_32.lib")  //Load ws2_32.dll

using namespace config4cpp;
using namespace std;
using namespace cv;
#define PI 3.14159265

// function forward declaration
Mat genTarTemplate(double R[], double alpha, double gama);
int SetBabyParameter(bool recalibrate_flag, int const babyIDnum, babyPara &currBaby, double R[], Mat refineDisk);
bool tracking(Point2d &dot_position, double &corr_out, double startX_bk, double startY_bk, Mat pf, Mat nf, Mat &templateImg, double R[], Mat const &refineDisk);
void myEllipseAngleEst(ellipsePara &ep, Mat const &currTarget, double R[]);
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
void removeReflection(Mat &eyeballregion_left_cal_3, Mat &eyeballregion_right_cal_3);
void irisSegmentation(Mat &eyeballregion_left_cal_3, Mat &eyeballregion_right_cal_3, Scalar skinAvg,		// Input
					  Mat &eyeballregion_left_bw, Mat &eyeballregion_right_bw);							// Output
int estGaze(Mat const &eyeballregion_left_bw, Mat const &eyeballregion_right_bw, Mat const &currFrame,
			 double X_eyeCenter_calL, double Y_eyeCenter_calL, 
			 double X_eyeCenter_calR, double Y_eyeCenter_calR,
			 babyPara const &currBaby);

bool DEBUG_showPlot=false;
bool DEBUG_log=true;
bool DEBUG_tartemplate=true;	// recompute TarTemplate gives bad accuracy on c++
int lost_track_only=0;

int babypool_temp[16]={28,30,31,32,34,35,36,37,38,39,40,41,42,43,44,45};
//int babypool_temp[16]={32};
std::vector<int> babypool (babypool_temp, babypool_temp + sizeof(babypool_temp) / sizeof(babypool_temp[0]) );

void main() {
// DEBUG
	string log_file_name;
	if (DEBUG_tartemplate)
		log_file_name="currTarget_th2_recomp_fix_";
	else 
		log_file_name="currTarget_th2_fix_";
// Initialize & Create socket	
    WSADATA wsaData;
	if(WSAStartup(MAKEWORD(2, 2), &wsaData)!=0)
	{
		std::cout<<"WSA Initialization failed!\r\n";
		WSACleanup();
		system("PAUSE");
	}
	SOCKET sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
	if(sock==INVALID_SOCKET)
	{
		std::cout<<"Socket creation failed.\r\n";
		WSACleanup();
		system("PAUSE");
	}


// Configure socket
    sockaddr_in sockAddr;	
    memset(&sockAddr, 0, sizeof(sockAddr));  //每个字节都用0填充
    sockAddr.sin_family = PF_INET;
    sockAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    sockAddr.sin_port = htons(11235);

// Connect to Server
	int connectSucces=-1;
	//while (connectSucces!=0) {
		connectSucces = connect(sock, (SOCKADDR*)&sockAddr, sizeof(SOCKADDR));
		//cout<<"Server not connected... "<<endl;
		//Sleep(1000);
	//}
	//cout<<"Connected to server !!"<<endl;
	// set to non-blocking mode
	u_long iMode=1;
	ioctlsocket(sock,FIONBIO,&iMode);
// Start console interface, select baby and landmark
	for (int babyi=0;babyi<babypool.size(); babyi++) {
		int babyIDnum=babypool[babyi];
	//cout<< "Please enter baby ID number "<<endl;
	//string userDecision;
	//getline(cin, userDecision);
	//int babyIDnum=stoi(userDecision);
	babyPara currBaby;
	//cout<< "Select facial landmarks manually ? (y/n) "<<endl;
	//getline(cin, userDecision);
	//while (  userDecision != "y" && userDecision != "n") {
	//			cout<< "Please enter 'y' or 'n' "<<endl;
	//			getline(cin, userDecision);
	//}
	//if (userDecision == "y")
	//	int calib_flag = SetBabyParameter(TRUE, babyIDnum, currBaby, R, refineDisk);
	//else
		int calib_flag = SetBabyParameter(FALSE, babyIDnum, currBaby, R, refineDisk);

// testing
	int N_trackFail=0;
	int N_eyeCrop=0;
	int N_estGaze=0;
	for (int video_i=0; video_i<currBaby.videoNames.size(); video_i++) {
		VideoCapture currVideo("F:/gkx/UIUC study/2016 spring/Ikids/Learning Data Collection/Target/"+to_string(babyIDnum)+"/"+currBaby.videoNames[video_i]);
		int totalFrame = currVideo.get(CV_CAP_PROP_FRAME_COUNT);
		cout<<"totalFrame = "<<totalFrame<<endl;
		
		Mat currFrame, preFrame;

		// initialization
		//currVideo.set(CV_CAP_PROP_POS_FRAMES,7000);
		bool success = currVideo.read(preFrame);
		if (!success)
			return;	
		cvtColor(preFrame, preFrame, CV_BGR2GRAY );
		////preFrame.convertTo(preFrame,CV_64FC1);
		preFrame.convertTo(preFrame,CV_32FC1);
		Point2d dot_position_pre(160, 120), dot_position(0.0,0.0);
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		Mat TarTemplate = genTarTemplate(R,0,0);
		//Mat TarTemplate_ori = genTarTemplate(R,0,0);
		while (1) {
			bool success = currVideo.read(currFrame);		//  currFrame= CV_8UC3
			if (!success)
				break;
			int currFrameNumber = currVideo.get(CV_CAP_PROP_POS_FRAMES);
			cout<<"currFrameNumber = "<<currFrameNumber<<endl;

			Mat currFrame_gray;
			cvtColor(currFrame, currFrame_gray, CV_BGR2GRAY );
			currFrame_gray.convertTo(currFrame_gray,CV_32FC1);
			// tracking
			double corr;
			success = tracking(dot_position, corr, dot_position_pre.x, dot_position_pre.y, preFrame, currFrame_gray, TarTemplate, R, refineDisk);
			if (!success) {			// tracking & template matching both fail
				cout<<"tracking fail, corr = "<<corr<<endl;
				N_trackFail++;
				if (DEBUG_showPlot) {
					namedWindow("currFrame",CV_WINDOW_AUTOSIZE);
					imshow("currFrame",currFrame);
					waitKey(1);
				}
					
				if (DEBUG_log) {
					FILE * (file);
					string logFile=log_file_name+to_string(babyIDnum)+".txt";
					file=fopen(logFile.c_str(),"a+");

					fprintf (file, "%d %d %s %s\n",video_i,currFrameNumber,"d","tracking_fail");

					fclose (file);
				}

				TarTemplate=genTarTemplate(R_ori,0,0);
				preFrame = currFrame_gray;
				dot_position_pre=dot_position;
				continue;
			}
			cout<<"corr = "<<corr<<endl;
			// ellipse fitting
			Mat currTarget;
			ellipsePara ep_curr;
			getRectSubPix(currFrame_gray, Size(2.0*R[4]+1.0,2.0*R[4]+1.0), dot_position, currTarget, -1);
			myEllipseAngleEst(ep_curr, currTarget, R);
			// Compute new target template
			if (DEBUG_tartemplate) {
				double Rfactor = max(ep_curr.EllipsePara[2], ep_curr.EllipsePara[3]) / R_ori[2];
				for (int i=0; i<=3; i++) {
					R[i] = R_ori[i] * Rfactor;
				}
				if (R[3] <= R_ori[4]-1 && R[2] > 0.6*R_ori[2]) {
					TarTemplate = genTarTemplate(R,ep_curr.alpha_est,ep_curr.gama_est);
				}
				else {
					for (int i=0; i<5; i++) {
						R[i] = R_ori[i] ;
					}
				}
			}
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
				N_eyeCrop++;
				if (DEBUG_showPlot) {
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
					namedWindow("left eye",CV_WINDOW_AUTOSIZE);
					namedWindow("right eye",CV_WINDOW_AUTOSIZE);
					imshow("currFrame",currFrame);
					imshow("left eye",eyeballregion_left_cal_3);
					imshow("right eye",eyeballregion_right_cal_3);
					waitKey(1);
				}

				if (DEBUG_log) {
					FILE * (file);
					string logFile=log_file_name+to_string(babyIDnum)+".txt";
					file=fopen(logFile.c_str(),"a+");

					fprintf (file, "%d %d %s %s\n",video_i,currFrameNumber,"d","eye_crop_fail");

					fclose (file);
				}

				preFrame = currFrame_gray;
				dot_position_pre=dot_position;
				continue;
			}
			// reflection removal
			removeReflection(eyeballregion_left_cal_3, eyeballregion_right_cal_3);
			// iris segmentation
			Scalar skinAvg = getSkinAvg(currFrame, dot_position);

			Mat eyeballregion_left_bw, eyeballregion_right_bw;
			irisSegmentation(eyeballregion_left_cal_3, eyeballregion_right_cal_3, skinAvg,		// Input
							 eyeballregion_left_bw, eyeballregion_right_bw);					// Output
			// estimate gaze direction
			int estDirection = estGaze(eyeballregion_left_bw, eyeballregion_right_bw, currFrame,
									 X_eyeCenter_calL, Y_eyeCenter_calL, 
									 X_eyeCenter_calR, Y_eyeCenter_calR,
									 currBaby);
			// debug
			if (estDirection==-1){
				currFrame.rowRange(Range::all()).colRange(Range(0,20))=255;
			}
			else if (estDirection==1) {
				currFrame.rowRange(Range::all()).colRange(Range(300,320))=255;
			}
			else {
				cout<<"gaze estimation failure, estDirection = "<<estDirection<<endl;
				N_estGaze++;
				if (DEBUG_showPlot) {
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
					namedWindow("left eye",CV_WINDOW_AUTOSIZE);
					namedWindow("right eye",CV_WINDOW_AUTOSIZE);
					namedWindow("eyeballregion_left_bw",CV_WINDOW_AUTOSIZE);
					namedWindow("eyeballregion_right_bw",CV_WINDOW_AUTOSIZE);
					imshow("currFrame",currFrame);
					eyeballregion_left_cal_3.convertTo(eyeballregion_left_cal_3,CV_8UC3);
					eyeballregion_right_cal_3.convertTo(eyeballregion_right_cal_3,CV_8UC3);
					imshow("left eye",eyeballregion_left_cal_3);
					imshow("right eye",eyeballregion_right_cal_3);
					imshow("eyeballregion_left_bw",255*eyeballregion_left_bw);
					imshow("eyeballregion_right_bw",255*eyeballregion_right_bw);
					waitKey(1);
				}

				if (DEBUG_log) {
					FILE * (file);
					string logFile=log_file_name+to_string(babyIDnum)+".txt";
					file=fopen(logFile.c_str(),"a+");

					fprintf (file, "%d %d %s %s\n",video_i,currFrameNumber,"d","gaze_est_fail");

					fclose (file);
				}

				preFrame = currFrame_gray;
				dot_position_pre=dot_position;
				continue;
			}
			
			if (DEBUG_showPlot) {
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

				circle(currFrame, p1, 1, Scalar(0,255,0), 1);
				circle(currFrame, p2, 1, Scalar(0,255,0), 1);
				circle(currFrame, p3, 1, Scalar(0,255,0), 1);
				circle(currFrame, p4, 1, Scalar(0,255,0), 1);

			
				namedWindow("currFrame",CV_WINDOW_AUTOSIZE);
				namedWindow("TarTemplate",CV_WINDOW_AUTOSIZE);
				namedWindow("left eye",CV_WINDOW_AUTOSIZE);
				namedWindow("right eye",CV_WINDOW_AUTOSIZE);
				namedWindow("eyeballregion_left_bw",CV_WINDOW_AUTOSIZE);
				namedWindow("eyeballregion_right_bw",CV_WINDOW_AUTOSIZE);
				imshow("currFrame",currFrame);
				imshow("TarTemplate",TarTemplate);
				eyeballregion_left_cal_3.convertTo(eyeballregion_left_cal_3,CV_8UC3);
				eyeballregion_right_cal_3.convertTo(eyeballregion_right_cal_3,CV_8UC3);

				resize( eyeballregion_left_cal_3,  eyeballregion_left_cal_3, Size(310,210));
				resize( eyeballregion_right_cal_3,  eyeballregion_right_cal_3, Size(310,210));
				imshow("left eye",eyeballregion_left_cal_3);
				imshow("right eye",eyeballregion_right_cal_3);
				imshow("eyeballregion_left_bw",255*eyeballregion_left_bw);
				imshow("eyeballregion_right_bw",255*eyeballregion_right_bw);
		
				waitKey(1);
			}
		
// Send estimated direction through TCP
			char* gaze_direction;
			if (estDirection==-1)
				gaze_direction="l";
			else if (estDirection==1)
				gaze_direction="r";

			else if (estDirection==0)
				gaze_direction="d";

			if (DEBUG_log) {
					FILE * (file);
					string logFile=log_file_name+to_string(babyIDnum)+".txt";
					file=fopen(logFile.c_str(),"a+");

					fprintf (file, "%d %d %s %s\n",video_i,currFrameNumber,gaze_direction,"success");

					fclose (file);
			}

			send(sock, gaze_direction, strlen(gaze_direction), 0);
			
			// update tracking information before processing next frame
			preFrame = currFrame_gray;
			dot_position_pre=dot_position;
		}	// while(1), go through all frames
		
	}	// go through all video files
	//cout<<"N_trackFail = "<<N_trackFail<<endl;
	//cout<<"N_eyeCrop = "<<N_eyeCrop<<endl;
	//cout<<"N_estGaze = "<<N_estGaze<<endl;
	//cout<<"lost_track_only = "<<lost_track_only<<endl;
	//system("pause");
	}

}