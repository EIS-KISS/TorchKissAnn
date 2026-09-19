add_executable(${PROJECT_NAME}_bench bench.cpp)
target_link_libraries(${PROJECT_NAME}_bench ${PROJECT_NAME}_common)
target_include_directories(${PROJECT_NAME}_bench PUBLIC ${COMMON_INCLUDE_DIRECTORYS} .)
set_target_properties(${PROJECT_NAME}_bench PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME}_bench REUSE_FROM ${PROJECT_NAME}_common)
target_compile_definitions(${PROJECT_NAME}_bench PRIVATE "_XOPEN_SOURCE")

install(TARGETS ${PROJECT_NAME}_bench RUNTIME DESTINATION bin)
