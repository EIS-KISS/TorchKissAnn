add_executable(${PROJECT_NAME}_train train.cpp)
target_link_libraries(${PROJECT_NAME}_train ${PROJECT_NAME}_common)
target_include_directories(${PROJECT_NAME}_train PUBLIC ${COMMON_INCLUDE_DIRECTORYS} .)
set_target_properties(${PROJECT_NAME}_train PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME}_train REUSE_FROM ${PROJECT_NAME}_common)
target_compile_definitions(${PROJECT_NAME}_train PRIVATE "_XOPEN_SOURCE")

install(TARGETS ${PROJECT_NAME}_train RUNTIME DESTINATION bin)
