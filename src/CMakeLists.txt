set(SRC_FILES
	data/generators/classextractordataset.cpp
	data/generators/eisgendata.cpp
	data/generators/trashgenerator.cpp
	data/loaders/mpt.cpp
	data/loaders/dirloader.cpp
	data/loaders/deduploader.cpp
	data/eistotorch.cpp
	data/print.cpp
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
	drt/base_funcs.cpp
	drt/NNLS.cpp
	drt/T2_inversion.cpp
	drt/drt.cpp
)

include_directories(utils .)

set(ENV{TORCH_INSTALL_PREFIX} "/usr")

find_package(Torch REQUIRED)
find_package(Armadillo REQUIRED)

set(COMMON_LINK_LIBRARIES ${TORCH_LIBRARIES} ${ARMADILLO_LIBRARIES} -leisgenerator)
set(COMMON_INCLUDE_DIRECTORYS ${TORCH_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})
set(COMMON_COMPILE_FLAGS "-std=c++20 -Wall -O2 -march=native -g -Wfatal-errors")

message("Linking: " "${COMMON_LINK_LIBRARIES}")
message("Includeing: " "${COMMON_INCLUDE_DIRECTORYS}")

add_library(${PROJECT_NAME}_common STATIC ${SRC_FILES})
target_link_libraries(${PROJECT_NAME}_common ${COMMON_LINK_LIBRARIES})
target_include_directories(${PROJECT_NAME}_common PUBLIC ${COMMON_INCLUDE_DIRECTORYS} .)
set_target_properties(${PROJECT_NAME}_common PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME}_common PRIVATE torchph.h)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_common)
target_include_directories(${PROJECT_NAME} PUBLIC ${COMMON_INCLUDE_DIRECTORYS} .)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME} REUSE_FROM ${PROJECT_NAME}_common)

add_executable(${PROJECT_NAME}_train train.cpp)
target_link_libraries(${PROJECT_NAME}_train ${PROJECT_NAME}_common)
target_include_directories(${PROJECT_NAME}_train PUBLIC ${COMMON_INCLUDE_DIRECTORYS} .)
set_target_properties(${PROJECT_NAME}_train PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME}_train REUSE_FROM ${PROJECT_NAME}_common)

add_executable(${PROJECT_NAME}_test test.cpp)
target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME}_common)
target_include_directories(${PROJECT_NAME}_test PUBLIC ${COMMON_INCLUDE_DIRECTORYS} .)
set_target_properties(${PROJECT_NAME}_test PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME}_test  REUSE_FROM ${PROJECT_NAME}_common)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(TARGETS ${PROJECT_NAME}_train RUNTIME DESTINATION bin)
