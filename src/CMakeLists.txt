set(SRC_FILES
	data/loaders/dirloader.cpp
	data/loaders/tarloader.cpp
	data/loaders/regressionloader.cpp
	data/loaders/regressiondirloader.cpp
	data/loaders/tardataset.cpp
	data/loaders/dirdataset.cpp
	data/loaders/eisspectradataset.cpp
	data/eistotorch.cpp
	data/print.cpp
	data/classextractordataset.cpp
	utils/tokenize.cpp
	utils/log.cpp
	utils/trainlog.cpp
	utils/randomgen.cpp
	utils/tensoroperators.cpp
	utils/microtar.cpp
	utils/modelscript.cpp
	utils/ploting.cpp
	utils/r2score.cpp
	utils/save.cpp
	gan/networks.cpp
	gan/simplenet.cpp
	gan/gan.cpp
	ann/simplenet.cpp
	ann/convnet.cpp
	ann/scriptnet.cpp
	ann/classification.cpp
	ann/regression.cpp
	ann/autoencoder.cpp
	loss/eisdistanceloss.cpp
	fit/fit.cpp
	globals.cpp
	net.cpp
)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/utils/gitrev.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/utils/gitrev.cpp" @ONLY)

list(APPEND SRC_FILES "${CMAKE_CURRENT_BINARY_DIR}/utils/gitrev.cpp")

include_directories(utils .)

add_library(${PROJECT_NAME}_common STATIC ${SRC_FILES})
target_link_libraries(${PROJECT_NAME}_common ${COMMON_LINK_LIBRARIES})
target_include_directories(${PROJECT_NAME}_common PUBLIC ${COMMON_INCLUDE_DIRECTORYS} .)
set_target_properties(${PROJECT_NAME}_common PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME}_common PRIVATE torchph.h)
target_compile_definitions(${PROJECT_NAME}_common PRIVATE "_XOPEN_SOURCE")

add_executable(${PROJECT_NAME}_utest test.cpp)
target_link_libraries(${PROJECT_NAME}_utest ${PROJECT_NAME}_common)
target_include_directories(${PROJECT_NAME}_utest PUBLIC ${COMMON_INCLUDE_DIRECTORYS} .)
set_target_properties(${PROJECT_NAME}_utest PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME}_utest  REUSE_FROM ${PROJECT_NAME}_common)
target_compile_definitions(${PROJECT_NAME}_utest PRIVATE "_XOPEN_SOURCE")

add_subdirectory(inference)
add_subdirectory(train)
add_subdirectory(test)
add_subdirectory(datasetinfo)
add_subdirectory(tune)
