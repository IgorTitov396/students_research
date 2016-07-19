project(runner)	

include_directories(../)

add_executable(runner runner.cpp)	

target_link_libraries(runner ${OpenCV_LIBS})

target_link_libraries(runner doc_tracker)	