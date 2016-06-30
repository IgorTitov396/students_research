project(doc_tracker)
find_package(OpenCV REQUIRED)
add_executable(doc_tracker main.cpp)
target_link_libraries(doc_tracker ${OpenCV_LIBS})