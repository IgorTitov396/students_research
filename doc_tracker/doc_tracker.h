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

void order_points(std::vector<cv::Point2f> &answer);

int is_detected(std::vector<cv::Point2f> my_points, std::vector<cv::Point2f> test_points, int frame_number, \
	int &total_frames, int &detected_frames);

std::vector<std::string> make_test_files(std::vector<std::string> data_files);

int trim_string(std::string &str);

void draw_contour(cv::Mat &image, std::vector<cv::Point2f> points);

int area_triangle(cv::Point2f point_1, cv::Point2f point_2, cv::Point2f point_3);

bool point_in_box(cv::Point2f point_1, cv::Point2f point_2, cv::Point2f point_3);

float distance(cv::Point2f point_1, cv::Point2f point_2);

std::vector<cv::Point2f> extreme_points(cv::Mat image);

bool is_square(std::vector<cv::Point2f> vector_of_points);

void my_algo(cv::Mat image, std::vector<cv::Point2f> &vector_of_points);

std::vector<cv::Point2f> calc_points(cv::Mat prev_img, cv::Mat next_img, std::vector<cv::Point2f> prev_pts);

int calc_test_points(std::vector<std::vector<cv::Point2f>> &test_points, std::string str);