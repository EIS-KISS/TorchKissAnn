cmake_minimum_required(VERSION 3.20)

project(torchkissann)

set(SRC_FILES
	main.cpp
	eisgeneratorloader.cpp
)

find_package(Torch REQUIRED)

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} -leisgenerator)
target_include_directories(${PROJECT_NAME} PUBLIC ${TORCH_INCLUDE_DIRS} .)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-std=c++20 -Wall -O3 -march=native -g" LINK_FLAGS "-flto")

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
