project(viewer)	

include_directories(../)

add_executable(viewer viewer.cpp)	

target_link_libraries(viewer ${OpenCV_LIBS})

target_link_libraries(viewer doc_tracker)