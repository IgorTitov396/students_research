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
	for(int i = 0; i < test_points.size(); i++) {
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

int main(int argc, char* argv[]) {
	int total_frames = 0, detected_frames = 0, is_paused = 0;
	if (argc != 2) {
		std::cout << "Wrong input parametrs" << std::endl;
		return 1;
	}
	std::ifstream input_file(argv[1]);
	if (!input_file.is_open()) {
		std::cout << "Can't open file" << std::endl;
		return 1;
	}
	std::string s_of_input_file;
	std::vector<std::string> input_data_files, test_data_files;
	while (std::getline(input_file, s_of_input_file)) {
		if (trim_string(s_of_input_file)) {
			std::cout << "Wrong input parametrs" << std::endl;
			return 1;
		}
		input_data_files.push_back(s_of_input_file);
	}
	if (input_data_files.size() == 0) {
		std::cout << "No input data" << std::endl;
		return 1;
	}
	test_data_files = make_test_files(input_data_files);
	for (int n = 0; n < input_data_files.size(); n++) {
		std::cout << "Name of file: " << input_data_files[n] << std::endl << std::endl;
		std::vector<std::vector<cv::Point2f>> test_points;
		std::vector<std::vector<cv::Point2f>> my_points;
		std::vector<cv::Mat> my_frames;
		if (calc_test_points(test_points, test_data_files[n])) {
			return 1;
		}
		cv::VideoCapture cap(input_data_files[n]);
		if (!cap.isOpened()) {
			if (argv[1] != NULL) {
				std::cout << "Can't open file: " << argv[1] << std::endl;
			}
			else {
				std::cout << "Wrong number of input parameters" << std::endl;
			}
			return 1;
		}

		cv::Mat next_frame, prev_frame, next_frame_show, prev_frame_show;
		if (!cap.read(prev_frame)) {
			std::cout << "Can't read first frame: " << argv[1] << std::endl;
			return 1;
		}
		std::vector<cv::Point2f> prev_points, next_points;
		my_algo(prev_frame, prev_points);
		if (prev_points.size() == 4) {
			draw_contour(prev_frame, prev_points, 5);
		}
		cv::namedWindow("window", cv::WINDOW_AUTOSIZE);
		cv::resize(prev_frame, prev_frame_show, cv::Size(prev_frame.cols / 2, prev_frame.rows / 2));
		int frame_number = 0;
		if (frame_number > test_points.size()) {
			std::cout << "Test frames less than video frames" << std::endl;
			return 1;
		}
		if (is_detected(prev_points, test_points[frame_number], frame_number, total_frames, detected_frames)) {
			std::cout << "Test points less than or more video points" << std::endl;
			return 1;
		}
		my_points.push_back(prev_points);
		my_frames.push_back(prev_frame);
		cv::imshow("window", prev_frame_show);
		char key = cvWaitKey(1);
		if (key == 27) {
			cv::destroyAllWindows();
			return 0;
		}
		if (key == 112) {
			while (key != 114) {
				key = cvWaitKey(1);
			}
		}
		for (;;) {
			if (my_points.size() == frame_number + 1) my_points.push_back(prev_points);
			if (my_frames.size() == frame_number + 1) my_frames.push_back(prev_frame);
			if (!cap.read(next_frame))
				break;
			if (prev_points.size() != 4) {
				my_algo(next_frame, next_points);
				if (next_points.size() == 4) {
					draw_contour(next_frame, next_points, 5);
				}
			}
			else {
				next_points = calc_points(prev_frame, next_frame, prev_points);
				draw_contour(next_frame, next_points, 5);
			}
			cv::resize(next_frame, next_frame_show, cv::Size(next_frame.cols / 2, next_frame.rows / 2));
			frame_number++;
			if (frame_number > test_points.size()) {
				std::cout << "Test frames less than video frames" << std::endl;
				return 1;
			}
			if (is_detected(next_points, test_points[frame_number], frame_number, total_frames, detected_frames)) {
				std::cout << "Test points less than or more video points" << std::endl;
				return 1;
			}
			cv::imshow("window", next_frame_show);
			key = cvWaitKey(1);
			if (key == 27)
				break;
			if (key == 112 || is_paused) {
				is_paused = 1;
				while (key != 114) {
					key = cvWaitKey(1);
					//b
					if (key == 98) {
						if (frame_number >= 2) {
							frame_number = frame_number - 2;
							//std::cout << frame_number << std::endl;
							next_points = my_points[frame_number + 1];
							cap.set(CV_CAP_PROP_POS_FRAMES, static_cast<float>(frame_number + 1));
							next_frame = my_frames.at(frame_number + 1).clone();
							break;
						}
					}
					//n
					if (key == 110) {
						int a = 0;
						break;
					}
				}
				if (key == 114) is_paused = 0;
			}
			prev_points = next_points;
			prev_frame = next_frame.clone();
		}
		cv::destroyAllWindows();
		std::cout << std::endl << "-----------------------------" << std::endl;
	}
	float accuracy = (static_cast<float>(detected_frames) / static_cast<float>(total_frames)) * 100;
	std::cout.precision(2);
	std::cout << std::endl << "Total frames : " << total_frames <<std::endl;
	std::cout << std::endl << "Frames detected : " << detected_frames << std::endl;
	std::cout << std::endl << "Accuracy : " << accuracy << "%" << std::endl;
	return 0;
}

