#include <opencv2\opencv.hpp>
#include "ellipsePara.h"
#include "decomposition.h"
using namespace cv;
#define PI 3.14159265

void myEllipseAngleEst(ellipsePara &ep, Mat const &currTarget, double R[]) {
	double x0 = R[4];	// should equal to 14+1 = 15 always since R[4] hard coded
	double y0 = R[4];
	Mat slice_idx(Mat::zeros(1,14,CV_32FC1)); // 14 = R_ori[4] = R[4], hard coded. CV_64FC1 makes 'cv::remap' memory exception
	for (int i=0;i<slice_idx.cols;i++) {
		slice_idx.at<float>(i) = i+1;
	}
	Mat targetRing_coor(Mat::zeros(2*360,2,CV_64FC1));

	int validMinMaxLoc=0;
	int targetRing_coor_idx=0;
	for (int i_angle=0;i_angle<360;i_angle+=5) {
		double curr_angle = PI*i_angle/180;
		Mat curr_x = x0+cos(curr_angle)*slice_idx;
		Mat curr_y = y0-sin(curr_angle)*slice_idx;
		Mat radial_sample;

		remap(currTarget, radial_sample, curr_x, curr_y,CV_INTER_LINEAR);
		radial_sample.convertTo(radial_sample,CV_64FC1);	// redundant. change to CV_32FC1 and <float> later !!!!!!!!!!!!!!!!!!!!!
		curr_x.convertTo(curr_x,CV_64FC1);
		curr_y.convertTo(curr_y,CV_64FC1);
		// find min first, then find max outside this min
		int temp_start = cvRound(R[1]);
		int temp_end = cvRound(R[3]);
		int ind_min, ind_max;
		Point minLoc, maxLoc;
		//try {
		minMaxLoc(radial_sample(Rect(temp_start,0,temp_end-temp_start+1,1)),NULL,NULL,&minLoc,NULL); //Rect(x,y,width,height)
		ind_min = cvRound(minLoc.x-1+R[1]);
		minMaxLoc(radial_sample(Rect(ind_min,0,temp_end-ind_min+1,1)),NULL,NULL,NULL,&maxLoc); //Rect(x,y,width,height)
		ind_max = cvRound(maxLoc.x-1+ind_min);
		//}
		//catch (...) {
		//	cout<<"BUGBUGBUGBUGBUGBUG"<<endl;
		//	cout<<"temp_start = "<<temp_start<<endl;
		//	cout<<"temp_end = "<<temp_end<<endl;
		//	system("pause");
		//}

		// check if max is outside of min
		if (ind_max>ind_min) {
			targetRing_coor.at<double>(targetRing_coor_idx,0) = curr_x.at<double>(0,ind_min);
			targetRing_coor.at<double>(targetRing_coor_idx,1) = curr_y.at<double>(0,ind_min);
			targetRing_coor.at<double>(targetRing_coor_idx+1,0) = curr_x.at<double>(0,ind_max);
			targetRing_coor.at<double>(targetRing_coor_idx+1,1) = curr_y.at<double>(0,ind_max);
			validMinMaxLoc++;
			targetRing_coor_idx+=2;
		}
	}
	validMinMaxLoc--;
	//try {
		Mat targetRing_coor_temp = targetRing_coor(Rect(0,0,2,2*validMinMaxLoc));
	//}
	//catch (...) {
	//		cout<<"BUGBUGBUGBUGBUGBUG 222222222"<<endl;
	//		system("pause");
	//}


	Mat targetRing_coor2 = targetRing_coor(Rect(0,0,2,2*validMinMaxLoc));
	if (validMinMaxLoc<5) {
		ep.alpha_est=0;
		ep.gama_est=0;
		ep.EllipsePara[0] = 0;
		ep.EllipsePara[1] = 0;
		ep.EllipsePara[2] = 0;
		ep.EllipsePara[3] = 0;
		ep.EllipsePara[4] = 0;
		ep.shape=-1;
		return;
	}
	// Ellipse fitting 
	Mat Ddata1, Ddata2;
	Mat tempMat1 = targetRing_coor2(Range::all(),Range(0,1));	//targetRing_coor2(:,1)
	Mat tempMat2 =  targetRing_coor2(Range::all(),Range(1,2)); //targetRing_coor2(:,2)

	Mat Ddata1array[]={tempMat1.mul(tempMat1),
					   tempMat1.mul(tempMat2),
					   tempMat2.mul(tempMat2)};
	cv::hconcat(Ddata1array,3,Ddata1);
	cv::hconcat(targetRing_coor2,Mat::ones(targetRing_coor2.rows,1,CV_64FC1),Ddata2);
	Mat Sdata1 = Ddata1.t() * Ddata1;
	Mat Sdata2 = Ddata1.t() * Ddata2;
	Mat Sdata3 = Ddata2.t() * Ddata2;
	Mat T = -Sdata3.inv() * Sdata2.t();
	Mat Mtemp = Sdata1 + Sdata2 * T;
	Mtemp.row(0)/=2;
	Mtemp.row(2)/=2;
	Mtemp.row(1)*=-1;
	Mat M(Mtemp.size(),Mtemp.type());
	Mtemp.row(0).copyTo(M.row(2));
	Mtemp.row(2).copyTo(M.row(0));
	Mtemp.row(1).copyTo(M.row(1));
	// M is non-symmetric matrix, so cannot use opencv eigen function
	EigenvalueDecomposition M_eig(M);
	Mat eigvec = M_eig.eigenvectors(); //need to normalize, CV64FC1
	for (int i=0;i<3;i++) {
		eigvec.col(i)/=norm(eigvec.col(i));
	}
	Mat cond_para =  4*eigvec.row(0).mul(eigvec.row(2))-eigvec.row(1).mul(eigvec.row(1));
	Mat a1(Mat::zeros(eigvec.rows,1,eigvec.type()));
	for (int i=0;i<3;i++) {
		if (cond_para.at<double>(i)>0) {
			eigvec.col(i).copyTo(a1.col(0));
			break;
		}
	}
	Mat parameter_estimated;
	cv::vconcat(a1,T*a1,parameter_estimated); // take the real part ????????????????????????????????????????????????????????????????????????

	double A = parameter_estimated.at<double>(0);
	double B=parameter_estimated.at<double>(1)/2;
	double C=parameter_estimated.at<double>(2);
	double D=parameter_estimated.at<double>(3)/2;
	double F=parameter_estimated.at<double>(4)/2;
	double G=parameter_estimated.at<double>(5);

	double zx = (C*D-B*F)/(B*B-A*C);
	double zy = (A*F-B*D)/(B*B-A*C);
	double aa_temp=sqrt(2*(A*F*F+C*D*D+G*B*B-2*B*D*F-A*C*G)/(B*B-A*C)/(sqrt((A-C)*(A-C)+4*B*B)-A-C));
    double bb_temp=sqrt(2*(A*F*F+C*D*D+G*B*B-2*B*D*F-A*C*G)/(B*B-A*C)/(-sqrt((A-C)*(A-C)+4*B*B)-A-C));

	if (aa_temp!=aa_temp || bb_temp!=bb_temp) {		// if 'aa_temp' or 'aa_temp' is complex number (regarded as NaN by c++)
		ep.alpha_est=0;
		ep.gama_est=0;
		ep.EllipsePara[0] = 0;
		ep.EllipsePara[1] = 0;
		ep.EllipsePara[2] = 0;
		ep.EllipsePara[3] = 0;
		ep.EllipsePara[4] = 0;
		ep.shape=-1;
		return;
	}

	double majorAxis=max(aa_temp,bb_temp);
    double minorAxis=min(aa_temp,bb_temp);

	double omega=0;
	if (B==0) {
		if (abs(A)<=abs(C)) {
			ep.shape=0;
			ep.EllipsePara[2] = majorAxis; // EllipsePara = EllipsePara=[zz[2],aa,bb,alpha]
			ep.EllipsePara[3] = minorAxis;
			omega=PI/2;
		}
		else {
			ep.shape=1;
			ep.EllipsePara[2] = minorAxis;
			ep.EllipsePara[3] = majorAxis;
			omega=0;
		}
		ep.EllipsePara[4] = 0;
	}
	else {
		if (abs(A)<=abs(C)) {
			ep.shape = 0;
			ep.EllipsePara[2] = majorAxis;
			ep.EllipsePara[3] = minorAxis;
			omega=PI/2-0.5*atan(2*B/(A-C));
		}
		else {
			ep.shape = 1;
			ep.EllipsePara[2] = minorAxis;
			ep.EllipsePara[3] = majorAxis;
			omega=-0.5*atan(2*B/(A-C));
		}
		ep.EllipsePara[4] = 0.5*atan(2*B/(A-C));	//acot(x)=atan(1/x)
	}
	ep.EllipsePara[0] = zx;
	ep.EllipsePara[1] = zy;
	
	double k = minorAxis/majorAxis;		// simplified from matlab if() statement ????????
	if (abs(omega)>PI/2)
		omega=omega-PI;
	if (omega<=0) {
		ep.gama_est = asin(-sqrt(((k*k-1)*cos(omega)*cos(omega))/((1-k*k)*sin(omega)*sin(omega)-1)));
		ep.alpha_est = asin(sqrt((1-k*k)*sin(omega)*sin(omega)));
	}
	else {
		ep.gama_est = asin(sqrt(((k*k-1)*cos(omega)*cos(omega))/((1-k*k)*sin(omega)*sin(omega)-1)));
		ep.alpha_est = asin(sqrt((1-k*k)*sin(omega)*sin(omega)));
	}

}
