cmake_minimum_required(VERSION 3.19)

project(torchkissann)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "/usr" CACHE PATH "..." FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set (CMAKE_CXX_STANDARD 20)

find_package(Torch REQUIRED)
find_package(tensorboard_logger REQUIRED)
find_package(Armadillo)

set(COMMON_LINK_LIBRARIES ${TORCH_LIBRARIES} ${ARMADILLO_LIBRARIES} eisgenerator)
set(COMMON_INCLUDE_DIRECTORYS ${TORCH_INCLUDE_DIRS})
set(COMMON_COMPILE_FLAGS "-Wall -O2 -march=native -g -Wfatal-errors")
message("Linking: " "${COMMON_LINK_LIBRARIES}")
message("Includeing: " "${COMMON_INCLUDE_DIRECTORYS}")

add_compile_definitions(MODEL_LISTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/modellists")

add_subdirectory(src)
