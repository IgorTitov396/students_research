#include "doc_tracker.h"

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

		cv::Mat next_frame, prev_frame, next_frame_copy, prev_frame_copy, next_frame_show, prev_frame_show;
		if (!cap.read(prev_frame)) {
			std::cout << "Can't read first frame: " << argv[1] << std::endl;
			return 1;
		}
		std::vector<cv::Point2f> prev_points, next_points;
		my_algo(prev_frame, prev_points);
		prev_frame_copy = prev_frame.clone();
		if (prev_points.size() == 4) {
			draw_contour(prev_frame_copy, prev_points);
		}
		cv::namedWindow("window", cv::WINDOW_AUTOSIZE);
		cv::resize(prev_frame_copy, prev_frame_show, cv::Size(prev_frame_copy.cols / 2, prev_frame_copy.rows / 2));
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
			next_frame_copy = next_frame.clone();
			if (prev_points.size() != 4) {
				my_algo(next_frame_copy, next_points);
				if (next_points.size() == 4) {
					draw_contour(next_frame_copy, next_points);
				}
			}
			else {
				next_points = calc_points(prev_frame, next_frame_copy, prev_points);
				draw_contour(next_frame_copy, next_points);
			}
			cv::resize(next_frame_copy, next_frame_show, cv::Size(next_frame_copy.cols / 2, next_frame_copy.rows / 2));
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
			if (key == 32 || is_paused) {
				if (key == 32 && !is_paused) key = 0;
				is_paused = 1;
				while (key != 32) {
					key = cvWaitKey(1);
					//b
					if (key == 97) {
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
					if (key == 100) {
						int a = 0;
						break;
					}
				}
				if (key == 32) is_paused = 0;
			}
			prev_points = next_points;
			prev_frame = next_frame.clone();
		}
		cv::destroyAllWindows();
		std::cout << std::endl << "-----------------------------" << std::endl;
	}
	float accuracy = (static_cast<float>(detected_frames) / static_cast<float>(total_frames)) * 100;
	std::cout.precision(2);
	std::cout << std::endl << "Total frames : " << total_frames << std::endl;
	std::cout << std::endl << "Frames detected : " << detected_frames << std::endl;
	std::cout << std::endl << "Accuracy : " << accuracy << "%" << std::endl;
	return 0;
}
