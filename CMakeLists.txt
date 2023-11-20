cmake_minimum_required(VERSION 3.26)

project(torchkissann LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "/usr" CACHE PATH "..." FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set (CMAKE_CXX_STANDARD 20)

find_package(Torch REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(sciplot REQUIRED)
pkg_search_module(JSONCPP REQUIRED jsoncpp)

include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)

if(NOT sciplot_FOUND)
	message(WARNING "Sciplot not found, application will be unable to generate plots")
else()
	add_definitions(-DENABLE_PLOTTING)
	message("Sciplot found, ploting enabled")
endif()

set(COMMON_LINK_LIBRARIES pthread ${JSONCPP_LIBRARIES} ${TORCH_LIBRARIES} eisgenerator eisdrt_torch)
set(COMMON_INCLUDE_DIRECTORYS ${JSONCPP_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
set(COMMON_COMPILE_FLAGS "-Wall -O2 -march=native -g -Wfatal-errors")
message("Linking: " "${COMMON_LINK_LIBRARIES}")
message("Includeing: " "${COMMON_INCLUDE_DIRECTORYS}")

add_compile_definitions(MODEL_LISTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/modellists")

add_subdirectory(src)
