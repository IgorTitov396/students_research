#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <math.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

static int epsilon = 20;

void draw_contour(cv::Mat &image, std::vector<cv::Point2f> points, int width) {
	
	for (int i = 0; i < points.size(); i++) {
		cv::line(image, points.at(i) , points.at((i + 1) % points.size()), cv::Scalar(0, 0, 255), width);
	}
}

int area_triangle(cv::Point2f point_1, cv::Point2f point_2, cv::Point2f point_3) {
	int value = (point_2.x - point_1.x) * (point_3.y - point_1.y) - (point_2.y - point_1.y)*(point_3.x - point_1.x);
	if (value > 0) return 1;
	if (value == 0) return 0;
	return -1;
}

bool point_in_box(cv::Point2f point_1, cv::Point2f point_2, cv::Point2f point_3) {
	return (point_1.x <= point_3.x && point_2.x >= point_3.x && point_1.y <= point_3.y && point_2.y >= point_3.y) || (point_1.x >= point_3.x && point_2.x <= point_3.x && point_1.y >= point_3.y && point_2.y <= point_3.y);
}

float distance(cv::Point2f point_1, cv::Point2f point_2) {
	return sqrt((point_2.x - point_1.x)*(point_2.x - point_1.x) + (point_2.y - point_1.y)*(point_2.y - point_1.y));
}


std::vector<cv::Point2f> extreme_points(cv::Mat image) {
	std::vector<cv::Point2f> points;
	for (int i = 0; i < image.cols; i++) {
		for (int j = 0; j < image.rows; j++) {
			if (image.at<float>(cv::Point2f(i, j)) == 255) {
				points.push_back(cv::Point2f(i, j));
			}
		}
	}
	size_t n = points.size();
	std::vector<cv::Point2f> answer;
	if (points.size() == 0) return answer;
	int first, q, next, i;
	int sign;
	first = 0;
	for (i = 1; i < n; ++i) {
		if (points[i].x < points[first].x || (points[i].x == points[first].x && points[i].y < points[first].y)) first = i;
	}
	q = first;
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

void my_algo(cv::Mat &image) {
	cv::Mat after_corner_Harris;
	cv::Mat gray;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::cornerHarris(gray, after_corner_Harris, 3, 3, 0.04);
	cv::threshold(after_corner_Harris, gray, 0.0001, 255, cv::THRESH_BINARY);
	std::vector<cv::Point2f> vector_of_points = extreme_points(gray);
	draw_contour(image, vector_of_points, 6);

}

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
	std::vector<cv::Point2f> vector_of_points = extreme_points(threshed);
	cv::Mat first_frame_copy = first_frame.clone();
	draw_contour(first_frame_copy, vector_of_points, 6);
	cv::namedWindow("Display window 5", CV_WINDOW_AUTOSIZE);
	cv::imshow("Display window 5", first_frame_copy);
	cv::waitKey(0);

	for (int i = 0; i < threshed.cols; i++) {
		for (int j = 0; j < threshed.rows; j++) {
			threshed.at<float>(cv::Point(i, j)) = 0;		
		}
	}
	for (int i = 0; i < vector_of_points.size(); i++) {
		threshed.at<float>(vector_of_points.at(i)) = 255;
	}

	cv::namedWindow("Display window 4", CV_WINDOW_AUTOSIZE);
	cv::imshow("Display window 4", threshed);
	cv::waitKey(0);
	/*
	cv::Mat color_gray, color_gray_frame, next_frame;
	cap.read(next_frame);
	std::vector<float> err;
	std::vector<uchar> status;
	cv::cvtColor(first_frame, color_gray, CV_BGR2GRAY);
	cv::cvtColor(next_frame, color_gray_frame, CV_BGR2GRAY);
	std::vector<cv::Point2f> vector_of_points_2(color_gray.cols * color_gray.rows);
	for (int i = 0; i < vector_of_points.size(); i++) {
		vector_of_points_2.at(i) = vector_of_points.at(i);
	}
	std::vector<cv::Point2f> vector_of_points_after;
	cv::calcOpticalFlowPyrLK(color_gray, color_gray_frame, vector_of_points, vector_of_points_after, status, err);	/*
	for (int i = 0; i < vector_of_points.size(); i++) {
		cv::Point2f current_point(vector_of_points.at(i).x, vector_of_points.at(i).y);
		prevPts.push_back(current_point);
	}
	*/
	cv::Mat frame;
	for (;;) {
		if (!cap.read(frame))
			break;
		
		my_algo(frame);
		cv::imshow("window", frame);
		char key = cvWaitKey(1);
		if (key == 27)
			break;
	}
	cv::destroyAllWindows();
	return 0;
}