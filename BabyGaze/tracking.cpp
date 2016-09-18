#include <opencv2\opencv.hpp>
//#include "programPara.h"		// cause "already defined" error. Because "programPara.h" already included in test.cpp, which calls tracking.cpp. So 
								// again including "programPara.h" here makes the variables in it get double defined. 
								// Note that the include guard does't work because each .cpp are compiled separately and then linked together. When compiling, 
								// each .cpp doesn't know each other, and they include "programPara.h" separately. However during linking, double define error occurs.
using namespace cv;
// function templates
Mat genTarTemplate(double R[], double alpha, double gama);
void refineDot(Mat const &nf, Point2d &dot_position, double const R[], Mat const &refineDisk);
extern double maxcorr_thresh_templateMatching;	// globa variable declaration
//string type2str(int type);
extern int lost_track_only;

bool tracking(Point2d &dot_position, double &corr_out, double startX_bk, double startY_bk, Mat pf, Mat nf, Mat &templateImg, double R[], Mat const &refineDisk) {
	// may be optimized by just passing in the 2*hws window region of pf and nf, or by constant reference
	// pos_corr={newX,newY,corr};
	double hws = R[4];
	double startX=2*hws+1;
    double startY=2*hws+1;
	Point2f center_bk(startX_bk,startY_bk);
	Point2f center(startX,startY); // Point2d (row,col)

	Mat pf_small, nf_small; // only contain the region around the target
	////pf.convertTo(pf,CV_32FC1);													//CV_64FC1 makes 'getRectSubPix' memory leak
	////nf.convertTo(nf,CV_32FC1);	
	getRectSubPix(pf, Size(4.0*hws+1.0,4.0*hws+1.0), center_bk, pf_small, -1);
	getRectSubPix(nf, Size(4.0*hws+1.0,4.0*hws+1.0), center_bk, nf_small, -1);
	//Mat Ix(pf_small.size(),CV_64FC1), Iy(pf_small.size(),CV_64FC1);
	Mat Ix, Iy;
	Scharr(pf_small,Ix,CV_32FC1,1,0,1,0,BORDER_DEFAULT );
	Scharr(pf_small,Iy,CV_32FC1,0,1,1,0,BORDER_DEFAULT );

	Mat G1 = templateImg - mean(templateImg);
	Mat G1_square = G1.mul(G1);
	////G1.convertTo(G1,CV_32FC1);
	////G1_square.convertTo(G1_square,CV_32FC1);

	Mat Ix_patch, Iy_patch, I_patch_pf, I_patch_nf;
	getRectSubPix(Ix, Size(2.0*hws+1.0,2.0*hws+1.0), center, Ix_patch, -1);
	getRectSubPix(Iy, Size(2.0*hws+1.0,2.0*hws+1.0), center, Iy_patch, -1);
	getRectSubPix(pf_small, Size(2.0*hws+1.0,2.0*hws+1.0), center, I_patch_pf, -1);
	getRectSubPix(nf_small, Size(2.0*hws+1.0,2.0*hws+1.0), center, I_patch_nf, -1);

	// iteration
	int ittN=0;
	double corr=0;
	double corr_pre=0;
	double corr_diff=1;

	while (corr_diff>1e-3 && ittN<=50) {
		if ((startX<=hws) || (startY<=hws) || (startX>pf_small.cols-hws) || (startY>pf_small.rows-hws)) {
			dot_position.x=0.0;
			dot_position.y=0.0;
			corr_out=0.0;
			return false;
		}
		Mat It_patch=I_patch_pf-I_patch_nf;

		if (!Ix_patch.isContinuous()) Ix_patch=Ix_patch.clone();// make them continuous in order
		if (!Iy_patch.isContinuous()) Iy_patch=Iy_patch.clone();// to reshape them
		if (!It_patch.isContinuous()) It_patch=It_patch.clone();// to reshape them

		Mat A,B,d;
		hconcat(Ix_patch.reshape(0,Ix_patch.total()), Iy_patch.reshape(0,Ix_patch.total()), A);
		B = It_patch.reshape(0,It_patch.total());

		solve(A,B,d,DECOMP_QR);

		float d0=d.at<float>(0);
		float d1=d.at<float>(1);

		startX=startX+d.at<float>(0);
        startY=startY+d.at<float>(1);
		getRectSubPix(nf_small, Size(2.0*hws+1.0,2.0*hws+1.0), Point2f(startX,startY), I_patch_nf, -1);

		Mat G2 = I_patch_nf - mean(I_patch_nf);
		Mat G2_square = G2.mul(G2);

		corr = sum(G1.mul(G2))[0] / sqrt(sum(G1_square)[0]*sum(G2_square)[0]);
		corr_diff = abs(corr_pre-corr);
		corr_pre=corr;

		ittN++;
	}
	dot_position.x = startX-2*hws-1+startX_bk;
	dot_position.y = startY-2*hws-1+startY_bk;
	corr_out = corr;
	
	// template matching if tracking fails, otherwise return TRUE
	if (corr_out<0.3 || corr_out!=corr_out) {
		//std::cout<<"corr = "<<corr<<", template matching instead !!!!"<<std::endl;
		Mat currFrame_gray2 = nf - mean(nf)[0];

		GaussianBlur(currFrame_gray2, currFrame_gray2, Size(5,5), 1.0, 1.0);
		Mat headBox, corrMap;
		int headBox_col=220;	// size of the head box
		int headBox_row=120;
		getRectSubPix(currFrame_gray2, Size(headBox_col,headBox_row), Point2d(159.0,119), headBox, -1);
		//getRectSubPix(currFrame_gray2, Size(220,100), Point2d(159.0,122.0), headBox, -1);
		//getRectSubPix(currFrame_gray2, Size(220,120), Point2d(159.0,132.0), headBox, -1);	// for baby 44, target out of head box, enlarge it helps with left accuracy.

		////Mat templateImg_CV_32FC1;
		////templateImg.convertTo(templateImg_CV_32FC1,CV_32FC1);
		////matchTemplate(headBox, templateImg_CV_32FC1, corrMap, CV_TM_CCORR_NORMED);
		matchTemplate(headBox, templateImg, corrMap, CV_TM_CCORR_NORMED);

		Point maxLoc;
		double maxcorr;
		minMaxLoc(corrMap, NULL, &maxcorr, NULL, &maxLoc);

		//dot_position.x=maxLoc.x-1+50+R[4];    // wrt currFrame
        //dot_position.y=maxLoc.y-1+73+R[4];
		dot_position.x=maxLoc.x+cvRound((320-headBox_col)/2.0)+R[4];    // wrt currFrame
        dot_position.y=maxLoc.y+cvRound((240-headBox_row)/2.0)+R[4];

		/*
		maxLoc.x=maxLoc.x+R[4];
		maxLoc.y=maxLoc.y+R[4];
		circle(headBox, maxLoc, 3, (0,255,0), 2);
		namedWindow("headBox",CV_WINDOW_AUTOSIZE);
		headBox.convertTo(headBox,CV_8UC1);
		imshow("headBox",headBox);
		waitKey(0);
		*/
		

		corr_out = maxcorr;
		if (maxcorr<maxcorr_thresh_templateMatching) {
			templateImg = genTarTemplate(R,0,0);
			corr_out = corr;
			return false;
		}

		//refineDot(nf, dot_position, R, refineDisk);	// 'refineDot()' not stable. Improve !!!!!!!!!!!!!!!!!!!!!!
		lost_track_only++;
		return true;
	}
	else {
		//refineDot(nf, dot_position, R, refineDisk);
		return true;
	}


}

// 'refineDot()' has internal linkage, only visible inside this file
static void refineDot(Mat const &nf, Point2d &dot_position, double const R[], Mat const &refineDisk) {
	double hws = cvRound(R[0]+3);
	Mat centerTargetRegion;	// centerTargetRegion = 9x9 matrix
							// refineDisk = 7x7
	getRectSubPix(nf, Size(2.0*hws+1.0,2.0*hws+1.0), dot_position, centerTargetRegion, -1);

	Mat matchMap;	// matchMap = 9x9 
	//matchTemplate(centerTargetRegion, refineDisk, matchMap, CV_TM_CCORR);	// how about 'CV_TM_CCOEFF' ????????
	matchTemplate(centerTargetRegion, refineDisk, matchMap, CV_TM_CCOEFF_NORMED);	// how about 'CV_TM_CCOEFF' ????????
	Point minLoc;
	minMaxLoc(matchMap, NULL, NULL, &minLoc, NULL);

	double offsetX = 3+minLoc.x - (centerTargetRegion.cols+1)/2;	// the 1st '3' is half size of 'refineDisk'
	double offsetY = 3+minLoc.y - (centerTargetRegion.rows+1)/2;
	dot_position.x += offsetX;
	dot_position.y += offsetY;
	/*
	centerTargetRegion.convertTo(centerTargetRegion,CV_8UC1);
	namedWindow("centerTargetRegion",CV_WINDOW_AUTOSIZE);
	imshow("centerTargetRegion",centerTargetRegion);
	waitKey(0);
	*/
}