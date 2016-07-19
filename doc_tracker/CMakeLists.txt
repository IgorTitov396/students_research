find_package(OpenCV REQUIRED)

project(doc_tracker)	

set(SOURCES "doc_tracker.cpp")

set(HEADERS "doc_tracker.h")

add_library(doc_tracker STATIC ${SOURCES})

target_link_libraries(doc_tracker ${OpenCV_LIBS})

add_subdirectory(runner)

add_subdirectory(viewer)