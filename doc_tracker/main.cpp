#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv\cv.h"
#include <opencv2\opencv.hpp>
#include "opencv\highgui.h"
#include <math.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

static int epsilon = 20;

int area_triangle(cv::Point point_1, cv::Point point_2, cv::Point point_3) {
	int value = (point_2.x - point_1.x) * (point_3.y - point_1.y) - (point_2.y - point_1.y)*(point_3.x - point_1.x);
	if (value > 0) return 1;
	if (value == 0) return 0;
	return -1;
}

bool point_in_box(cv::Point point_1, cv::Point point_2, cv::Point point_3) {
	return (point_1.x <= point_3.x && point_2.x >= point_3.x && point_1.y <= point_3.y && point_2.y >= point_3.y) || (point_1.x >= point_3.x && point_2.x <= point_3.x && point_1.y >= point_3.y && point_2.y <= point_3.y);
}

float distance(cv::Point point_1, cv::Point point_2) {
	return sqrt((point_2.x - point_1.x)*(point_2.x - point_1.x) + (point_2.y - point_1.y)*(point_2.y - point_1.y));
}


std::vector<cv::Point2f> extreme_points(cv::Mat image) {
	std::vector<cv::Point> points;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (image.at<float>(i, j) == 255) {
				points.push_back(cv::Point(i, j));
			}
		}
	}
	size_t n = points.size();
	int first, q, next, i;
	int sign;
	first = 0;
 	for (i = 1; i < n; ++i) {
		if (points[i].x < points[first].x || (points[i].x == points[first].x && points[i].y < points[first].y)) first = i;
	}
	q = first;
	std::vector<cv::Point2f> answer;
	do {
		answer.push_back(points.at(q));
		next = q;
		for (i = n - 1; i >= 0; --i)
			//if (points[i].x != points[q].x || points[i].y != points[q].y)
			if ((points[i].x != points[q].x || points[i].y != points[q].y) && (distance(points[i], points[q]) > epsilon)) {
				sign = area_triangle(points[q], points[i], points[next]);

				if (next == q || sign > 0 || (sign == 0 && point_in_box(points[next], points[q], points[i])))
					next = i;
			}
		q = next;
	//} while (q != first);
	} while (distance(points[q], points[first]) > epsilon);
	return answer;
}

using std::vector;

int main(int argc, char* argv[]) {
	cv::VideoCapture cap(argv[1]); 
	if (!cap.isOpened()) {
		if (argv[1] != NULL) {
			std::cout << "Can't open file: " << argv[1] << std::endl;
		}
		else {
			std::cout << "Wrong number of input parameters" << std::endl;
		}
	return -1;
	}

	cv::Mat first_frame;
	//cap.retrieve(first_frame);
	if (!cap.read(first_frame)) {
		std::cout << "Can't read first frame: " << argv[1] << std::endl;
		return 1;
	}
	cv::imshow("first frame", first_frame);

	cv::Mat after_corner_Harris;
	cv::Mat gray;
	cv::Mat threshed;
	cv::cvtColor(first_frame, gray, CV_BGR2GRAY);
	cv::cornerHarris(gray, after_corner_Harris, 3, 3, 0.04);
	cv::threshold(after_corner_Harris, threshed, 0.0001, 255, cv::THRESH_BINARY);
	cv::namedWindow("Display window 3", CV_WINDOW_AUTOSIZE);
	cv::imshow("Display window 3", threshed);
	vector<cv::Point2f> vector_of_points = extreme_points(threshed);

	for (int i = 0; i < threshed.rows; i++) {
		for (int j = 0; j < threshed.cols; j++) {
			threshed.at<float>(i, j) = 0;
		}
	}
	for (int i = 0; i < vector_of_points.size(); i++) {
		threshed.at<float>(vector_of_points.at(i).x, vector_of_points.at(i).y) = 255;
	}

	std::vector<cv::Point2f> prevPts;
	cv::Mat color_gray;
	cv::cvtColor(first_frame, color_gray, CV_BGR2GRAY);
	for (int i = 0; i < vector_of_points.size(); i++) {
		cv::Point2f current_point(vector_of_points.at(i).x, vector_of_points.at(i).y);
		prevPts.push_back(current_point);
	}
	cv::namedWindow("Display window 4", CV_WINDOW_AUTOSIZE);
	cv::imshow("Display window 4", threshed);
	cv::waitKey(0);
	cv::Mat prev_image = first_frame.clone();
	std::vector<cv::Point2f> vector_of_points_after;
	cv::Mat frame;
	for (;;) {
		if (!cap.read(frame))
		break;
		cv::Mat color_gray_frame;
		cv::cvtColor(frame, color_gray_frame, CV_BGR2GRAY);
		std::vector<float> err;
		std::vector<uchar> status;
		cv::Size winSize(31, 31);
		cv::calcOpticalFlowPyrLK(color_gray, color_gray_frame, prevPts, vector_of_points_after, status, err, winSize, 3);
		prev_image = frame.clone();
		for (int i = 0; i < frame.rows; i++) {
			for (int j = 0; j < frame.cols; j++) {
				frame.at<cv::Vec3b>(i, j)[0] = 0;
				frame.at<cv::Vec3b>(i, j)[1] = 0;
				frame.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}
		cv::imshow("window", frame);
		char key = cvWaitKey(33);
		if (key == 27) 
		break;
	}
	cv::destroyAllWindows();
	return 0;
}