cmake_minimum_required(VERSION 3.20)

project(torchkissann)

set(SRC_FILES
	eisgendata.cpp
	testgenerator.cpp
	net.cpp
	log.cpp
	tokenize.cpp
	eistotorch.cpp
)

find_package(Torch REQUIRED)

add_executable(${PROJECT_NAME} ${SRC_FILES} main.cpp)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} -leisgenerator)
target_include_directories(${PROJECT_NAME} PUBLIC ${TORCH_INCLUDE_DIRS} .)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-std=c++20 -Wall -O2 -march=native -g" LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME} PRIVATE torchph.h)

add_executable(${PROJECT_NAME}_train ${SRC_FILES} train.cpp)
target_link_libraries(${PROJECT_NAME}_train ${TORCH_LIBRARIES} -leisgenerator)
target_include_directories(${PROJECT_NAME}_train PUBLIC ${TORCH_INCLUDE_DIRS} .)
set_target_properties(${PROJECT_NAME}_train PROPERTIES COMPILE_FLAGS "-std=c++20 -Wall -O2 -march=native -g" LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME}_train  REUSE_FROM ${PROJECT_NAME})

#add_executable(${PROJECT_NAME}_test ${SRC_FILES} test.cpp)
#target_link_libraries(${PROJECT_NAME}_test ${TORCH_LIBRARIES} -leisgenerator)
#target_include_directories(${PROJECT_NAME}_test PUBLIC ${TORCH_INCLUDE_DIRS} .)
#set_target_properties(${PROJECT_NAME}_test PROPERTIES COMPILE_FLAGS "-std=c++20 -Wall -O3 -march=native -g" LINK_FLAGS "-flto")

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
