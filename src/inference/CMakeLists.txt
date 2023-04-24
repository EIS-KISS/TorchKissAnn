add_executable(${PROJECT_NAME} inference.cpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_common)
target_include_directories(${PROJECT_NAME} PUBLIC ${COMMON_INCLUDE_DIRECTORYS} .)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME} REUSE_FROM ${PROJECT_NAME}_common)
target_compile_definitions(${PROJECT_NAME} PRIVATE "_XOPEN_SOURCE")

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
