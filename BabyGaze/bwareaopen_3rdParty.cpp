#include <opencv2\opencv.hpp>
using namespace cv;

Mat threshSegments(Mat &src, Mat const &src_gray, double threshSize) {
    // FindContours:
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat srcBuffer, output;
    src.copyTo(srcBuffer);
    findContours(srcBuffer, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_TC89_KCOS);

    vector<vector<Point> > allSegments;

    // For each segment:
	vector<double> tempB;
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::drawContours(srcBuffer, contours, i, Scalar(200, 0, 0), 1, 8, hierarchy, 0, Point());
        cv::Rect brect = cv::boundingRect(contours[i]);
        cv::rectangle(srcBuffer, brect, Scalar(255, 0, 0));

        int result;
        vector<Point> segment;
        for (unsigned int row = brect.y; row < brect.y + brect.height; ++row) {
            for (unsigned int col = brect.x; col < brect.x + brect.width; ++col) {
                result = pointPolygonTest(contours[i], Point(col, row), false);
                if (result == 1 || result == 0) {
                    segment.push_back(Point(col, row));
                }
            }
        }
        allSegments.push_back(segment);
    }

    output = Mat::zeros(src.size(), CV_32FC1);
	vector<vector<Point>> validSegment;
    for (int segmentCount = 0; segmentCount < allSegments.size(); ++segmentCount) {
        vector<Point> segment = allSegments[segmentCount];
        if(segment.size() > threshSize){					// Future: add a upper bound on the segmentation size ?????
			validSegment.push_back(segment);
			/*
            for (int idx = 0; idx < segment.size(); ++idx) {
                output.at<float>(segment[idx].y, segment[idx].x) = 1;
            }
			*/
        }
    }
	// KEVIN - keep the segment with minumum mean pixel value
	if (validSegment.size()>1) {
		vector<double> tempB;
		for (int i=0; i<validSegment.size(); i++) {
			double tempAvg=0.0;
			vector<Point2i> currPixels = validSegment[i];
			for (int pIdx=0; pIdx<currPixels.size(); pIdx++) {
				tempAvg += src_gray.at<float>(currPixels[pIdx].y, currPixels[pIdx].x);
			}
			tempAvg /= currPixels.size();
			tempB.push_back(tempAvg);
		}
		int idxB = min_element(tempB.begin(), tempB.end()) - tempB.begin();
		double valB = tempB[idxB];
		if (valB < 0.5*mean(src_gray)[0]) {
			vector<Point2i> bestPixels = validSegment[idxB];
			for (int pIdx=0; pIdx<bestPixels.size(); pIdx++) {
				output.at<float>(bestPixels[pIdx].y, bestPixels[pIdx].x) = 1;
			}
		}
	}
	else if (validSegment.size() ==1 ) {		// it is possible that there is no segments (all small segments killed by size thresholding)
		for (int idx = 0; idx < validSegment[0].size(); ++idx) {
             output.at<float>(validSegment[0][idx].y, validSegment[0][idx].x) = 1;
        }
	}
	//
	
    return output;
}