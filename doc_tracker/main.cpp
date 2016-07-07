#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <math.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <exception>

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

void my_algo(cv::Mat &image, std::vector<cv::Point2f> &vector_of_points) {
	cv::Mat after_corner_Harris;
	cv::Mat gray;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::cornerHarris(gray, after_corner_Harris, 3, 3, 0.04);
	cv::threshold(after_corner_Harris, gray, 0.0001, 255, cv::THRESH_BINARY);
	vector_of_points = extreme_points(gray);
	draw_contour(image, vector_of_points, 5);

}

std::vector<cv::Point2f> calc_points(cv::Mat prev_img, cv::Mat next_img, std::vector<cv::Point2f> prev_pts)
{
	std::vector<cv::Point2f> next_pts;
	cv::Mat prev_gray, next_gray;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::cvtColor(prev_img, prev_gray, CV_RGB2GRAY);
	cv::cvtColor(next_img, next_gray, CV_RGB2GRAY);
	cv::calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, next_pts, status, err);
	for (int i = 0; i < prev_pts.size(); i++) {
		if (status[i] == 0) next_pts[i] = prev_pts[i];
	}
	return next_pts;
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

	cv::Mat first_frame, next_frame, prev_frame;
	//cap.retrieve(first_frame);
	if (!cap.read(first_frame)) {
		std::cout << "Can't read first frame: " << argv[1] << std::endl;
		return 1;
	}
	cv::imshow("first frame", first_frame);
	std::vector<cv::Point2f> prev_points, next_points;
	prev_frame = first_frame.clone();
	my_algo(first_frame, prev_points);
	cv::imshow("window", first_frame);
	for (;;) {
		if (!cap.read(next_frame))
			break;
		next_points = calc_points(prev_frame, next_frame, prev_points);
		draw_contour(next_frame, next_points, 5);
		cv::imshow("window", next_frame);
		char key = cvWaitKey(1);
		if (key == 27)
			break;
		prev_points = next_points;
		prev_frame = next_frame.clone();
	}
	cv::destroyAllWindows();
	return 0;
}