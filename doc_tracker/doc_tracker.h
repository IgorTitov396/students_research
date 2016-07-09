#pragma once
#pragma once
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
#include <string>

static float height_coef = 0.15;
static float width_coef = 0.15;
static int epsilon = 20;
static int test_epsilon = 30;
static float coef = 0.4;

void order_points(std::vector<cv::Point2f> &answer) {
	cv::Point2f min = answer.at(0);
	for (int i = 1; i < answer.size(); i++) {
		if (answer[i].x + answer[i].y < min.x + min.y) min = answer[i];
	}
	cv::Point2f buf;
	while (answer[1] != min) {
		buf = answer[0];
		for (int i = 1; i < answer.size(); i++) {
			answer[i - 1] = answer[i];
		}
		answer[answer.size() - 1] = buf;
	}
}

int is_detected(std::vector<cv::Point2f> my_points, std::vector<cv::Point2f> test_points, int frame_number, int &total_frames, int &detected_frames) {
	if (my_points.size() == 0) {
		std::cout << "Frame :" << frame_number + 1 << " NOT DETECTED" << std::endl;
		total_frames++;
		return 0;
	}
	if (my_points.size() != test_points.size()) return 1;
	for (int i = 0; i < test_points.size(); i++) {
		if (std::pow(my_points[i].x - test_points[i].x, 2) + std::pow(my_points[i].y - test_points[i].y, 2) > std::pow(test_epsilon, 2)) {
			std::cout << "Frame :" << frame_number + 1 << " NOT DETECTED" << std::endl;
			total_frames++;
			return 0;
		}
	}
	std::cout << "Frame :" << frame_number + 1 << " DETECTED" << std::endl;
	total_frames++;
	detected_frames++;
	return 0;
}

std::vector<std::string> make_test_files(std::vector<std::string> data_files) {
	std::vector<std::string> test_files;
	std::string buf;
	int pos_point;
	for (int i = 0; i < data_files.size(); i++) {
		pos_point = data_files[i].find('.');
		if (pos_point == -1) {
			buf = data_files[i] + ".gt.xml";
		}
		else {
			buf = data_files[i].substr(0, pos_point) + ".gt.xml";
		}
		test_files.push_back(buf);
	}
	return test_files;
}

int trim_string(std::string &str) {
	for (int i = 0; i < str.size(); i++) {
		if (str[i] != ' ') {
			str = str.substr(i, str.size() - i + 1);
			break;
		}
		if (i == str.size() - 1) return 1;
	}
	for (int i = str.size() - 1; i >= 0; i--) {
		if (str[i] != ' ') {
			str = str.substr(0, i + 1);
			break;
		}
		if (i == 0) return 1;
	}
	return 0;
}

void draw_contour(cv::Mat &image, std::vector<cv::Point2f> points, int width) {

	for (int i = 0; i < points.size(); i++) {
		cv::line(image, points.at(i), points.at((i + 1) % points.size()), cv::Scalar(0, 0, 255), width);
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
	for (int i = static_cast<int>(image.cols * width_coef); i < static_cast<int>(image.cols * (1 - width_coef)); i++) {
		for (int j = static_cast<int>(image.rows * height_coef); j < static_cast<int>(image.rows * (1 - height_coef)); j++) {
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

bool is_square(std::vector<cv::Point2f> vector_of_points) {
	if (vector_of_points.size() != 4) return false;
	for (int i = 0; i < vector_of_points.size(); i++) {
		float a, b, c1, c2;
		a = std::pow(vector_of_points.at((i + 1) % 4).x - vector_of_points.at((i) % 4).x, 2) + std::pow(vector_of_points.at((i + 1) % 4).y - vector_of_points.at((i) % 4).y, 2);
		b = std::pow(vector_of_points.at((i + 2) % 4).x - vector_of_points.at((i + 1) % 4).x, 2) + std::pow(vector_of_points.at((i + 2) % 4).y - vector_of_points.at((i + 1) % 4).y, 2);
		c1 = (std::pow(vector_of_points.at((i + 2) % 4).x - vector_of_points.at((i) % 4).x, 2) + std::pow(vector_of_points.at((i + 2) % 4).y - vector_of_points.at((i) % 4).y, 2))  * (1 - coef);
		c2 = (std::pow(vector_of_points.at((i + 2) % 4).x - vector_of_points.at((i) % 4).x, 2) + std::pow(vector_of_points.at((i + 2) % 4).y - vector_of_points.at((i) % 4).y, 2))  * (1 + coef);
		if ((c1 >= a + b) || (c2 <= a + b)) return false;
	}
	return true;
}

void my_algo(cv::Mat image, std::vector<cv::Point2f> &vector_of_points) {
	std::vector<cv::Point2f> free_vector;
	cv::Mat after_corner_Harris;
	cv::Mat gray;
	float kernel[9] = { -0.1, -0.1, -0.1, -0.1, 1.5, -0.1, -0.1, -0.1, -0.1 };
	cv::Mat kernel_matrix = cv::Mat(3, 3, CV_32FC1, kernel);
	cv::filter2D(image, image, -1, kernel_matrix);
	float kernel2[9] = { -0.1, -0.1, -0.1, -0.1, 2.4, -0.1, -0.1, -0.1, -0.1 };
	cv::Mat kernel_matrix2 = cv::Mat(3, 3, CV_32FC1, kernel2);
	cv::filter2D(image, image, -1, kernel_matrix2);
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::cornerHarris(gray, after_corner_Harris, 11, 9, 0.04);
	cv::threshold(after_corner_Harris, gray, 4, 255, cv::THRESH_BINARY);
	vector_of_points = extreme_points(gray);
	bool flag = is_square(vector_of_points);
	if ((vector_of_points.size() != 4 || !flag)) vector_of_points = free_vector;
	if (vector_of_points.size() == 4) order_points(vector_of_points);
}

std::vector<cv::Point2f> calc_points(cv::Mat prev_img, cv::Mat next_img, std::vector<cv::Point2f> prev_pts)
{
	std::vector<cv::Point2f> next_pts;
	cv::Mat prev_gray, next_gray;
	std::vector<uchar> status;
	std::vector<float> err;
	float kernel[9] = { -0.1, -0.1, -0.1, -0.1, 2.0, -0.1, -0.1, -0.1, -0.1 };
	cv::Mat kernel_matrix = cv::Mat(3, 3, CV_32FC1, kernel);
	//cv::filter2D(prev_img, prev_img, -1, kernel_matrix);
	cv::filter2D(next_img, next_img, -1, kernel_matrix);
	cv::cvtColor(prev_img, prev_gray, CV_RGB2GRAY);
	cv::cvtColor(next_img, next_gray, CV_RGB2GRAY);
	cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 200, 0.0003);
	cv::calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, next_pts, status, err, cv::Size(120, 120), 3, termcrit, 0, 0.001);
	for (int i = 0; i < prev_pts.size(); i++) {
		if (status[i] == 0) next_pts[i] = prev_pts[i];
	}
	return next_pts;
}

int calc_test_points(std::vector<std::vector<cv::Point2f>> &test_points, std::string str) {
	if (test_points.size() != 0) test_points.resize(0);
	std::ifstream file(str);
	if (!file.is_open()) {
		std::cout << "Can't open file" << std::endl;
		return 1;
	}
	std::string s;
	int index = 0, x_pos, y_pos, end_pos;
	std::string x_string, y_string;
	std::vector<cv::Point2f> buffer(4);
	while (std::getline(file, s)) {

		if ((x_pos = s.find("\" x=\"")) == -1) continue;
		if ((y_pos = s.find("\" y=\"")) == -1) {
			std::cout << "Wrong test parametrs" << std::endl;
			return 1;
		}
		if ((end_pos = s.find("\"/>")) == -1) {
			std::cout << "Wrong test parametrs" << std::endl;
			return 1;
		}
		x_string = s.substr(x_pos + 5, y_pos - x_pos + 4);
		y_string = s.substr(y_pos + 5, y_pos - end_pos + 4);
		buffer[index] = cv::Point2f(std::stof(x_string), std::stof(y_string));
		index++;
		if (index / 4 > 0) {
			index = 0;
			test_points.push_back(buffer);
		}
	}
	if (index != 0) {
		std::cout << "Wrong test parametrs" << std::endl;
		return 1;
	}
	return 0;
}