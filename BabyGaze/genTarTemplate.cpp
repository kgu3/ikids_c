
#include <opencv2\opencv.hpp>
using namespace cv;
using namespace std;
Mat genTarTemplate(double R[], double alpha, double gama) {	// 'alpha' and 'gama' should be in radians
	
	double R_large[5];
	for (int i=0;i<5;i++) {
		R_large[i]=R[i]*5;
	}

	int tempSize=1+2*R_large[4];
	int tempSize_small=1+2*R[4];
	
	////Mat TarTemplate(Mat::ones(tempSize_small,tempSize_small,CV_64FC1));
	////Mat TarTemplate_next_center_big(Mat::ones(tempSize,tempSize,CV_64FC1));
	////Mat TarTemplate_next_center(Mat::ones(tempSize_small,tempSize_small,CV_64FC1));
	Mat TarTemplate(Mat::ones(tempSize_small,tempSize_small,CV_32FC1));
	Mat TarTemplate_next_center_big(Mat::ones(tempSize,tempSize,CV_32FC1));
	Mat TarTemplate_next_center(Mat::ones(tempSize_small,tempSize_small,CV_32FC1));

	// 1st - compute center template, then reisze it to small
	for (int rIdx=0;rIdx<tempSize;rIdx++) {
		for (int cIdx=0;cIdx<tempSize;cIdx++) {
			double currX = cIdx - R_large[4];
			double currY = rIdx - R_large[4];
			double currR = sqrt(currX*currX+currY*currY);

			if ((currR <= R_large[0]) || (currR <= R_large[2] && currR > R_large[1])) {
				////TarTemplate_next_center_big.at<double>(rIdx,cIdx) = -1;
				TarTemplate_next_center_big.at<float>(rIdx,cIdx) = -1;
			}
			else if (currR > R_large[3]) {
				////TarTemplate_next_center_big.at<double>(rIdx,cIdx) = 0;
				TarTemplate_next_center_big.at<float>(rIdx,cIdx) = 0;
			}
		}
	}
	resize(TarTemplate_next_center_big,TarTemplate_next_center,TarTemplate_next_center.size(),0,0,INTER_AREA);
	// 2nd - compute target template rotated by alpha (vertical) & gama (hozirontal)
	////Mat TarSlice(Mat::zeros(1,R[4]+1+15,CV_64FC1));
	Mat TarSlice(Mat::zeros(1,R[4]+1+15,CV_32FC1));
	Mat aux = TarSlice.colRange(0,R[4]+1).rowRange(0,1);
	TarTemplate_next_center(Range(R[4],R[4]+1),Range(R[4],tempSize_small)).copyTo(aux);
	for (int rIdx=0;rIdx<tempSize_small;rIdx++) {
		for (int cIdx=0;cIdx<tempSize_small;cIdx++) {
			double currX = cIdx - R[4];
			double currY = rIdx - R[4];
			double currX_c = currX/cos(gama)-tan(alpha)*tan(gama)*currY;
			double currY_c = currY/cos(alpha);
			int TarSliceIdx = cvRound(sqrt(currX_c*currX_c+currY_c*currY_c));

			if (TarSliceIdx >= TarSlice.cols)			// deal with the points outside of 29x29 range
				TarTemplate.at<float>(rIdx,cIdx) = 0;
			else 
				////TarTemplate.at<double>(rIdx,cIdx) = TarSlice.at<double>(TarSliceIdx);
				TarTemplate.at<float>(rIdx,cIdx) = TarSlice.at<float>(TarSliceIdx);
			
		}
	}
	return TarTemplate;
}


/*
	FILE * (file);
	file=fopen("inputmatriks.txt","w");
	for (int i=0;i<29;i++) {
		for (int j=0;j<29;j++) {
			double currval=TarTemplate_next_center.at<double>(i,j);
			fprintf (file, "%1.3f, ",currval);
		}
		fprintf (file,"\n");
	}
	fclose (file);
*/