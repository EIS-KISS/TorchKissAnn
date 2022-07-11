set(SRC_FILES
	data/generators/classextractordataset.cpp
	data/generators/eisgendata.cpp
	data/generators/testgenerator.cpp
	data/generators/trashgenerator.cpp
	data/loaders/csv.cpp
	data/loaders/mpt.cpp
	data/loaders/dirloader.cpp
	data/eistotorch.cpp
	data/print.cpp
	svm/svmloader.cpp
	svm/svm.cpp
	utils/tokenize.cpp
	utils/log.cpp
	utils/randomgen.cpp
	utils/losslog.cpp
	utils/tensoroperators.cpp
	gan/networks.cpp
	gan/simplenet.cpp
	gan/gan.cpp
	ann/net.cpp
	ann/ann.cpp
	nearest/nearest.cpp
	drt/drt.cpp
)

include_directories(utils .)

set(ENV{TORCH_INSTALL_PREFIX} "/usr")

find_package(Torch REQUIRED)

add_executable(${PROJECT_NAME} ${SRC_FILES} main.cpp)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} -leisgenerator -lsvm)
target_include_directories(${PROJECT_NAME} PUBLIC ${TORCH_INCLUDE_DIRS} .)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-std=c++20 -Wall -O2 -march=native -g -Wfatal-errors" LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME} PRIVATE torchph.h)

add_executable(${PROJECT_NAME}_train ${SRC_FILES} train.cpp)
target_link_libraries(${PROJECT_NAME}_train ${TORCH_LIBRARIES} -leisgenerator -lsvm)
target_include_directories(${PROJECT_NAME}_train PUBLIC ${TORCH_INCLUDE_DIRS} .)
set_target_properties(${PROJECT_NAME}_train PROPERTIES COMPILE_FLAGS "-std=c++20 -Wall -O2 -march=native -g -Wfatal-errors" LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME}_train  REUSE_FROM ${PROJECT_NAME})

add_executable(${PROJECT_NAME}_test ${SRC_FILES} test.cpp)
target_link_libraries(${PROJECT_NAME}_test ${TORCH_LIBRARIES} -leisgenerator -lsvm)
target_include_directories(${PROJECT_NAME}_test PUBLIC ${TORCH_INCLUDE_DIRS} .)
set_target_properties(${PROJECT_NAME}_test PROPERTIES COMPILE_FLAGS "-std=c++20 -Wall -O2 -march=native -g -Wfatal-errors" LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME}_test  REUSE_FROM ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(TARGETS ${PROJECT_NAME}_train RUNTIME DESTINATION bin)
