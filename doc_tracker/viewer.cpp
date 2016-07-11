#include "doc_tracker.h"

int main(int argc, char* argv[]) {
	if (argc != 2) {
		std::cout << "Wrong input parametrs" << std::endl;
		return 1;
	}
	cv::VideoCapture cap(argv[1]);
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
	cv::imshow("window", prev_frame_show);
	char key = cvWaitKey(1);
	if (key == 27) {
		cv::destroyAllWindows();
		return 0;
	}
	for (;;) {
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
		cv::imshow("window", next_frame_show);
		key = cvWaitKey(1);
		if (key == 27)
			break;
		prev_points = next_points;
		prev_frame = next_frame.clone();
	}
	cv::destroyAllWindows();
	return 0;
}
