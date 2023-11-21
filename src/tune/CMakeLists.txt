add_executable(${PROJECT_NAME}_tune tune.cpp)
target_link_libraries(${PROJECT_NAME}_tune ${PROJECT_NAME}_common)
target_include_directories(${PROJECT_NAME}_tune PUBLIC ${COMMON_INCLUDE_DIRECTORYS} .)
set_target_properties(${PROJECT_NAME}_tune PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME}_tune REUSE_FROM ${PROJECT_NAME}_common)
target_compile_definitions(${PROJECT_NAME}_tune PRIVATE "_XOPEN_SOURCE")

install(TARGETS ${PROJECT_NAME}_tune RUNTIME DESTINATION bin)
