project(doc_tracker)
find_package(OpenCV REQUIRED)
add_executable(doc_tracker runner.cpp viewer.cpp doc_tracker.h)
target_link_libraries(doc_tracker ${OpenCV_LIBS})