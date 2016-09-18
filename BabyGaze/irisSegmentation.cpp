#include <opencv2\opencv.hpp>
using namespace cv;

extern double RegionThresh;
extern double MinIrisSize;
Mat threshSegments(Mat &src, Mat const &src_gray, double threshSize);
void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs);
string type2str(int type);

void irisSegmentation(Mat &eyeballregion_left_cal_3, Mat &eyeballregion_right_cal_3, Scalar skinAvg,		// Input
					  Mat &eyeballregion_left_bw, Mat &eyeballregion_right_bw)							// Output
{
	// Left Eye
	Mat eyeballregion_left_gray, eyeballregion_left;
	cvtColor(eyeballregion_left_cal_3, eyeballregion_left_gray, CV_BGR2GRAY);
	eyeballregion_left_gray.convertTo(eyeballregion_left_gray, CV_32FC1);
	eyeballregion_left_cal_3.convertTo(eyeballregion_left_cal_3,CV_64FC3);	// 'skinAvg' is CV_64FC4
	subtract(eyeballregion_left_cal_3, skinAvg, eyeballregion_left);		// 'subtract' only works with CV_64F ???
	eyeballregion_left = abs(eyeballregion_left);
	eyeballregion_left.convertTo(eyeballregion_left,CV_8UC3);
	cvtColor(eyeballregion_left,eyeballregion_left,CV_BGR2GRAY);

	eyeballregion_left.convertTo(eyeballregion_left,CV_32FC1);
	divide(eyeballregion_left, eyeballregion_left_gray, eyeballregion_left);
	//threshold(eyeballregion_left, eyeballregion_left, 5.0, 2.0, THRESH_TRUNC);	// currently eyeballregion_left(eyeballregion_left>5)=5, not same as Matlab !!!!!!!!!!!!!!!!!!!!!!!
Mat eyeballregion_left_temp1,eyeballregion_left_temp2;
threshold(eyeballregion_left, eyeballregion_left_temp1, 5.0, 2.0, THRESH_BINARY);
threshold(eyeballregion_left, eyeballregion_left_temp2, 5.0, 2.0, THRESH_TOZERO_INV);
eyeballregion_left=eyeballregion_left_temp1+eyeballregion_left_temp2;

	GaussianBlur(eyeballregion_left, eyeballregion_left, Size(5,5), 0.5,0.5);
	medianBlur(eyeballregion_left, eyeballregion_left, 3);
	//Mat vtempL = eyeballregion_left.reshape(0,1);		// Wrong. reshape() doesn't copy data, so sort vtempL actually sorts eyeballregion_left.
	Mat vtempL;
	eyeballregion_left.copyTo(vtempL);
	vtempL=vtempL.reshape(0,1);
	sort(vtempL, vtempL, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
	double totalEL = sum(vtempL)[0];

	// vector<Mat> tempbw_vecL;					// tempbw_vecL.push_back(tempLayerL) doesn't work, because 'tempLayerL' is not data copied, only header copied. Need to use tempbw_vecL.push_back(tempLayerL.clone());
	Mat tempbwL(Mat::zeros(eyeballregion_left_gray.size(),CV_32FC1));
	Mat tempLayerL;
	for (double perct=45.0; perct<=90.0; perct+=5.0) {
		double irisEnergy = 0;
		double irisEnergyRatio = 0;
		int i = 0;
		while (irisEnergyRatio < perct / 100.0) {
			i++;
			irisEnergy += vtempL.at<float>(i);
			irisEnergyRatio=irisEnergy/totalEL;
		}
		threshold(eyeballregion_left, tempLayerL, vtempL.at<float>(i), 1, THRESH_BINARY);
		//tempbw_vecL.push_back(tempLayerL.clone());
		add(tempbwL, tempLayerL, tempbwL);
	}
	threshold(tempbwL, eyeballregion_left_bw, RegionThresh, 1, THRESH_BINARY);
	eyeballregion_left_bw.convertTo(eyeballregion_left_bw,CV_8UC1);
	eyeballregion_left_bw = threshSegments(eyeballregion_left_bw, eyeballregion_left_gray, MinIrisSize);	// 3rd party code http://stackoverflow.com/questions/19732431/how-to-filter-small-segments-from-image-in-opencv?rq=1  !!!!!!!!!!!!!!!

	// Right Eye
	Mat eyeballregion_right_gray, eyeballregion_right;
	cvtColor(eyeballregion_right_cal_3, eyeballregion_right_gray, CV_BGR2GRAY );
	eyeballregion_right_gray.convertTo(eyeballregion_right_gray, CV_32FC1);
	eyeballregion_right_cal_3.convertTo(eyeballregion_right_cal_3,CV_64FC1);
	subtract(eyeballregion_right_cal_3, skinAvg, eyeballregion_right);
	eyeballregion_right=abs(eyeballregion_right);
	eyeballregion_right.convertTo(eyeballregion_right,CV_8UC3);
	cvtColor(eyeballregion_right,eyeballregion_right,CV_BGR2GRAY);
	eyeballregion_right.convertTo(eyeballregion_right,CV_32FC1);
	divide(eyeballregion_right, eyeballregion_right_gray, eyeballregion_right);
	//threshold(eyeballregion_right, eyeballregion_right, 5.0, 2.0, THRESH_TRUNC);	// currently eyeballregion_left(eyeballregion_left>5)=5, not same as Matlab !!!!!!!!!!!!!!!!!!!!!!!
Mat eyeballregion_right_temp1,eyeballregion_right_temp2;
threshold(eyeballregion_right, eyeballregion_right_temp1, 5.0, 2.0, THRESH_BINARY);
threshold(eyeballregion_right, eyeballregion_right_temp2, 5.0, 2.0, THRESH_TOZERO_INV);
eyeballregion_right=eyeballregion_right_temp1+eyeballregion_right_temp2;

	GaussianBlur(eyeballregion_right, eyeballregion_right, Size(5,5), 0.5,0.5);
	medianBlur(eyeballregion_right, eyeballregion_right, 3);
	Mat vtempR;
	eyeballregion_right.copyTo(vtempR);
	vtempR = vtempR.reshape(0,1);
	sort(vtempR, vtempR, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
	double totalER = sum(vtempR)[0];

	Mat tempbwR(Mat::zeros(eyeballregion_right_gray.size(),CV_32FC1));
	Mat tempLayerR;
	for (double perct=45.0; perct<=90.0; perct+=5.0) {
		double irisEnergy = 0;
		double irisEnergyRatio = 0;
		int i = 0;
		while (irisEnergyRatio < perct / 100.0) {
			i++;
			irisEnergy += vtempR.at<float>(i);
			irisEnergyRatio=irisEnergy/totalER;
		}
		threshold(eyeballregion_right, tempLayerR, vtempR.at<float>(i), 1, THRESH_BINARY);
		add(tempbwR, tempLayerR, tempbwR);
	}
	threshold(tempbwR, eyeballregion_right_bw, RegionThresh, 1, THRESH_BINARY);
	eyeballregion_right_bw.convertTo(eyeballregion_right_bw,CV_8UC1);
	eyeballregion_right_bw = threshSegments(eyeballregion_right_bw, eyeballregion_right_gray, MinIrisSize);	// 3rd party code http://stackoverflow.com/questions/19732431/how-to-filter-small-segments-from-image-in-opencv?rq=1  !!!!!!!!!!!!!!!

}

/*
	vector<vector<Point2i>> CCL;
	FindBlobs(eyeballregion_left_bw, CCL);
	if (CCL.size()>1) {
		vector<double> tempB;
		for (int i=0; i<CCL.size(); i++) {
			double tempAvg=0.0;
			vector<Point2i> currPixels = CCL[i];
			for (int pIdx=0; pIdx<currPixels.size(); pIdx++) {
				tempAvg += eyeballregion_left_gray.at<float>(currPixels[pIdx].y, currPixels[pIdx].x);
			}
			tempAvg /= currPixels.size();
			tempB.push_back(tempAvg);
		}
		int idxB = min_element(tempB.begin(), tempB.end()) - tempB.begin();
		double valB = tempB[idxB];
		eyeballregion_left_bw.setTo(0.0);
		if (valB < 0.5*mean(eyeballregion_left_gray)[0]) {
			vector<Point2i> bestPixels = CCL[idxB];
			for (int pIdx=0; pIdx<bestPixels.size(); pIdx++) {
				eyeballregion_left_bw.at<float>(bestPixels[pIdx].y, bestPixels[pIdx].x) = 1;
			}
		}

	}
*/

/*
	vector<vector<Point2i>> CCR;
	FindBlobs(eyeballregion_right_bw, CCR);
	if (CCR.size()>1) {
		vector<double> tempB;
		for (int i=0; i<CCR.size(); i++) {
			double tempAvg=0.0;
			vector<Point2i> currPixels = CCR[i];
			for (int pIdx=0; pIdx<currPixels.size(); pIdx++) {
				tempAvg += eyeballregion_right_gray.at<float>(currPixels[pIdx].y, currPixels[pIdx].x);
			}
			tempAvg /= currPixels.size();
			tempB.push_back(tempAvg);
		}
		int idxB = min_element(tempB.begin(), tempB.end()) - tempB.begin();
		double valB = tempB[idxB];
		eyeballregion_right_bw.setTo(0.0);
		if (valB < 0.5*mean(eyeballregion_right_gray)[0]) {
			vector<Point2i> bestPixels = CCR[idxB];
			for (int pIdx=0; pIdx<bestPixels.size(); pIdx++) {
				eyeballregion_right_bw.at<float>(bestPixels[pIdx].y, bestPixels[pIdx].x) = 1;
			}
		}

	}
*/