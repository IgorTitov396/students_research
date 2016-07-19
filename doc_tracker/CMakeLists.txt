find_package(OpenCV REQUIRED)

set(SOURCES "../doc_tracker.cpp")

set(HEADERS "../doc_tracker.h")

add_subdirectory(runner)

add_subdirectory(viewer)