
#include <opencv2\opencv.hpp>
using namespace cv;

void estInPlaneRot(Mat const &Ra, Mat const &Rr, vector<Point2f> const &xyo_curr, 
									 vector<Point2f> const &xy_curr,
									 vector<Point2f> const &xy_init, 
									 vector<Point2f> const &xyo_init, Mat &T_curr4) {
	//Mat Wtemp(Mat::eye(8,8,CV_32FC1));
	// LS rotation matrix solution
	Mat temp1 = Mat(xy_init).reshape(1).t();
	Mat temp2 = Mat(xyo_init).reshape(1).t();
	Mat temp3(Mat::zeros(3,8,CV_32FC1));
	hconcat(temp1,temp2,temp3.rowRange(0,2).colRange(Range::all()));
	 
	Mat A = Ra * Rr * temp3;
	A = A.rowRange(0,2).colRange(Range::all());

	temp1 = Mat(xy_curr).reshape(1);
	temp2 = Mat(xyo_curr).reshape(1);
	vconcat(temp1,temp2,temp3);
	Mat Stemp = A * temp3;

	SVD svd_rot;
	Mat uS,vST,wS;
	svd_rot.compute(Stemp,wS,uS,vST);

	T_curr4 = Mat::eye(3,3,CV_32FC1);
	T_curr4.rowRange(0,2).colRange(0,2) = vST.t() * uS.t();
}
