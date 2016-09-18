
// Class 'babyPara' contains the parameters for each baby, along some necessary methods.
#ifndef __babyPara_H_INCLUDED__   // if x.h hasn't been included yet...
#define __babyPara_H_INCLUDED__	  // Note: header guard only prevent a header from being included more than once into a single including file, not from being included one time into multiple different code files.

#include <string>
#include <opencv2\opencv.hpp>
#include <config4cpp/Configuration.h>
#include <string> 
#include <vector>

using namespace std;
using namespace config4cpp;
using namespace cv;

class babyPara {
public:
	// declare baby variables 
	// from .cfg file
	string fid_state;
	vector<string> videoNames;
	//string vidObj_1;	// for debugging only !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	double half_eyeball_H_ori;
	double half_eyeball_V_ori;
	double UnityTimer;
	double VideoLogTimeDiff_1;	// need VideoLogTimeDiff_2 later !!!!!!!!!!!!!!!!!!!!

	cv::Point2d dot_position;
	double centerX_left;
	double centerY_left;
	double centerX_right;
	double centerY_right;

	vector<cv::Point2f>  xy_init;
	vector<cv::Point2f>  xyo_init;

	// computed
	double X_eyeCenter_offsetL;	// For EyeCenter Calib, used to calculate iris boundary w.r.t currFrame
	double Y_eyeCenter_offsetL;
	double X_eyeCenter_offsetR;
	double Y_eyeCenter_offsetR;

	double X_ul_offsetL;	//For Whole-EyeRegion Calib
	double Y_ul_offsetL;
	double X_br_offsetL;
	double Y_br_offsetL;
	double X_ul_offsetR;
	double Y_ul_offsetR;
	double X_br_offsetR;
	double Y_br_offsetR;

	Mat coorL_all_xyz, coorR_all_xyz;
	
	// functions
	void populatePara(int babyIDnum) {
		string babyIDtemp = "baby_" + to_string(babyIDnum);
		const char*  babyID= babyIDtemp.c_str();

		Configuration * cfg = Configuration::create();
		//cv::String vidObj_1;
		try{	
			cfg->parse("C:/Users/kevin/Documents/Visual Studio 2012/Projects/BabyGaze/BabyGaze/babyProfile.cfg");
			fid_state = cfg->lookupString(babyID, "fid_state");
			int Nvideo=1;
			while (1) {
				string temp = "videoName_" + to_string(Nvideo);
				string currVideoName = cfg->lookupString(babyID, &temp[0],"NULL");
				if (currVideoName.compare("NULL")) {
					videoNames.push_back(currVideoName);
					Nvideo++;
				}
				else
					break;
			}

			//vidObj_1 = cfg->lookupString(babyID, "videoName_6","NULL");
			half_eyeball_H_ori = cfg->lookupFloat(babyID, "half_eyeball_H_ori");
			half_eyeball_V_ori = cfg->lookupFloat(babyID, "half_eyeball_V_ori");
			UnityTimer = cfg->lookupFloat(babyID, "UnityTimer");
			VideoLogTimeDiff_1 = cfg->lookupFloat(babyID, "VideoLogTimeDiff_1");

			StringVector dotP;
			cfg->lookupList(babyID, "dot_position", dotP);
			dot_position.x = atof(dotP[1]);
			dot_position.y = atof(dotP[0]);

		    centerX_left =  cfg->lookupFloat(babyID, "centerX_left");
			centerY_left =  cfg->lookupFloat(babyID, "centerY_left");
			centerX_right =  cfg->lookupFloat(babyID, "centerX_right");
			centerY_right =  cfg->lookupFloat(babyID, "centerY_right");

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
	
		} catch(const ConfigurationException & ex) {
			cout << ex.c_str() << endl;
		}


		X_eyeCenter_offsetL=centerX_left-dot_position.x;
		Y_eyeCenter_offsetL=centerY_left-dot_position.y;
		X_eyeCenter_offsetR=centerX_right-dot_position.x;
		Y_eyeCenter_offsetR=centerY_right-dot_position.y;

		X_ul_offsetL=centerX_left-half_eyeball_H_ori-dot_position.x;    // Target Center + this = Upper left corner of eye region (when 
		Y_ul_offsetL=centerY_left-half_eyeball_V_ori-dot_position.y;    // there is no head rotation)
		X_br_offsetL=centerX_left+half_eyeball_H_ori-dot_position.x;
		Y_br_offsetL=centerY_left+half_eyeball_V_ori-dot_position.y;
		X_ul_offsetR=centerX_right-half_eyeball_H_ori-dot_position.x;    // Target Center + this = Upper left corner of eye region (when 
		Y_ul_offsetR=centerY_right-half_eyeball_V_ori-dot_position.y;    // there is no head rotation)
		X_br_offsetR=centerX_right+half_eyeball_H_ori-dot_position.x; 
		Y_br_offsetR=centerY_right+half_eyeball_V_ori-dot_position.y;
		// Left Eye
		Mat coorL_all_x(Mat::zeros(1,2*half_eyeball_H_ori+1,CV_32FC1));
		Mat coorL_all_y(Mat::zeros(2*half_eyeball_V_ori+1,1,CV_32FC1));
		for (double i=0; i<2*half_eyeball_H_ori+1; i++) {
			coorL_all_x.at<float>(i) = X_ul_offsetL+i;
		}

		for (double i=0; i<2*half_eyeball_V_ori+1; i++) {
			coorL_all_y.at<float>(i) = Y_ul_offsetL+i;
		}

		cv::repeat(coorL_all_x, 2*half_eyeball_V_ori+1, 1, coorL_all_x);
		cv::repeat(coorL_all_y, 1, 2*half_eyeball_H_ori+1, coorL_all_y);

		coorL_all_x = coorL_all_x.reshape(0,1);
		coorL_all_y = coorL_all_y.reshape(0,1);
		cv::vconcat( coorL_all_x, coorL_all_y, coorL_all_xyz);
		cv::vconcat( coorL_all_xyz, Mat::zeros(1,coorL_all_xyz.cols,CV_32FC1), coorL_all_xyz);
		// Right Eye
		Mat coorR_all_x(Mat::zeros(1,2*half_eyeball_H_ori+1,CV_32FC1));
		Mat coorR_all_y(Mat::zeros(2*half_eyeball_V_ori+1,1,CV_32FC1));
		for (double i=0; i<2*half_eyeball_H_ori+1; i++) {
			coorR_all_x.at<float>(i) = X_ul_offsetR+i;
		}

		for (double i=0; i<2*half_eyeball_V_ori+1; i++) {
			coorR_all_y.at<float>(i) = Y_ul_offsetR+i;
		}

		cv::repeat(coorR_all_x, 2*half_eyeball_V_ori+1, 1, coorR_all_x);
		cv::repeat(coorR_all_y, 1, 2*half_eyeball_H_ori+1, coorR_all_y);

		coorR_all_x = coorR_all_x.reshape(0,1);
		coorR_all_y = coorR_all_y.reshape(0,1);
		cv::vconcat( coorR_all_x, coorR_all_y, coorR_all_xyz);
		cv::vconcat( coorR_all_xyz, Mat::zeros(1,coorR_all_xyz.cols,CV_32FC1), coorR_all_xyz);
	}
	
	void reComputePara() {
		X_eyeCenter_offsetL=centerX_left-dot_position.x;
		Y_eyeCenter_offsetL=centerY_left-dot_position.y;
		X_eyeCenter_offsetR=centerX_right-dot_position.x;
		Y_eyeCenter_offsetR=centerY_right-dot_position.y;

		X_ul_offsetL=centerX_left-half_eyeball_H_ori-dot_position.x;    // Target Center + this = Upper left corner of eye region (when 
		Y_ul_offsetL=centerY_left-half_eyeball_V_ori-dot_position.y;    // there is no head rotation)
		X_br_offsetL=centerX_left+half_eyeball_H_ori-dot_position.x;
		Y_br_offsetL=centerY_left+half_eyeball_V_ori-dot_position.y;
		X_ul_offsetR=centerX_right-half_eyeball_H_ori-dot_position.x;    // Target Center + this = Upper left corner of eye region (when 
		Y_ul_offsetR=centerY_right-half_eyeball_V_ori-dot_position.y;    // there is no head rotation)
		X_br_offsetR=centerX_right+half_eyeball_H_ori-dot_position.x; 
		Y_br_offsetR=centerY_right+half_eyeball_V_ori-dot_position.y;
		// Left Eye
		Mat coorL_all_x(Mat::zeros(1,2*half_eyeball_H_ori+1,CV_32FC1));
		Mat coorL_all_y(Mat::zeros(2*half_eyeball_V_ori+1,1,CV_32FC1));
		for (double i=0; i<2*half_eyeball_H_ori+1; i++) {
			coorL_all_x.at<float>(i) = X_ul_offsetL+i;
		}

		for (double i=0; i<2*half_eyeball_V_ori+1; i++) {
			coorL_all_y.at<float>(i) = Y_ul_offsetL+i;
		}

		cv::repeat(coorL_all_x, 2*half_eyeball_V_ori+1, 1, coorL_all_x);
		cv::repeat(coorL_all_y, 1, 2*half_eyeball_H_ori+1, coorL_all_y);

		coorL_all_x = coorL_all_x.reshape(0,1);
		coorL_all_y = coorL_all_y.reshape(0,1);
		cv::vconcat( coorL_all_x, coorL_all_y, coorL_all_xyz);
		cv::vconcat( coorL_all_xyz, Mat::zeros(1,coorL_all_xyz.cols,CV_32FC1), coorL_all_xyz);
		// Right Eye
		Mat coorR_all_x(Mat::zeros(1,2*half_eyeball_H_ori+1,CV_32FC1));
		Mat coorR_all_y(Mat::zeros(2*half_eyeball_V_ori+1,1,CV_32FC1));
		for (double i=0; i<2*half_eyeball_H_ori+1; i++) {
			coorR_all_x.at<float>(i) = X_ul_offsetR+i;
		}

		for (double i=0; i<2*half_eyeball_V_ori+1; i++) {
			coorR_all_y.at<float>(i) = Y_ul_offsetR+i;
		}

		cv::repeat(coorR_all_x, 2*half_eyeball_V_ori+1, 1, coorR_all_x);
		cv::repeat(coorR_all_y, 1, 2*half_eyeball_H_ori+1, coorR_all_y);

		coorR_all_x = coorR_all_x.reshape(0,1);
		coorR_all_y = coorR_all_y.reshape(0,1);
		cv::vconcat( coorR_all_x, coorR_all_y, coorR_all_xyz);
		cv::vconcat( coorR_all_xyz, Mat::zeros(1,coorR_all_xyz.cols,CV_32FC1), coorR_all_xyz);
	}


};

#endif 
