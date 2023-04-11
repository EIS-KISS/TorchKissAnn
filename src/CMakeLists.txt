set(SRC_FILES
	data/generators/classextractordataset.cpp
	data/generators/eisgendata.cpp
	data/generators/trashgenerator.cpp
	data/generators/parameterregressiondataset.cpp
	data/loaders/mpt.cpp
	data/loaders/dirloader.cpp
	data/loaders/deduploader.cpp
	data/eistotorch.cpp
	data/print.cpp
	utils/tokenize.cpp
	utils/log.cpp
	utils/trainlog.cpp
	utils/randomgen.cpp
	utils/tensoroperators.cpp
	gan/networks.cpp
	gan/simplenet.cpp
	gan/gan.cpp
	ann/net.cpp
	ann/classification.cpp
	ann/regression.cpp
	nearest/nearest.cpp
)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/utils/gitrev.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/utils/gitrev.cpp" @ONLY)

list(APPEND SRC_FILES "${CMAKE_CURRENT_BINARY_DIR}/utils/gitrev.cpp")

include_directories(utils .)

add_library(${PROJECT_NAME}_common STATIC ${SRC_FILES})
target_link_libraries(${PROJECT_NAME}_common ${COMMON_LINK_LIBRARIES})
target_include_directories(${PROJECT_NAME}_common PUBLIC ${COMMON_INCLUDE_DIRECTORYS} .)
set_target_properties(${PROJECT_NAME}_common PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME}_common PRIVATE torchph.h)
target_compile_definitions(${PROJECT_NAME}_common PRIVATE DRT_ENABLED)
target_compile_definitions(${PROJECT_NAME}_common PRIVATE "_XOPEN_SOURCE")

add_executable(${PROJECT_NAME}_test test.cpp)
target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME}_common)
target_include_directories(${PROJECT_NAME}_test PUBLIC ${COMMON_INCLUDE_DIRECTORYS} .)
set_target_properties(${PROJECT_NAME}_test PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME}_test  REUSE_FROM ${PROJECT_NAME}_common)
target_compile_definitions(${PROJECT_NAME}_test PRIVATE DRT_ENABLED)
target_compile_definitions(${PROJECT_NAME}_test PRIVATE "_XOPEN_SOURCE")

add_subdirectory(inference)
add_subdirectory(train)
add_subdirectory(datasetinfo)
