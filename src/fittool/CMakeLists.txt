add_executable(${PROJECT_NAME}_fit fittool.cpp)
target_link_libraries(${PROJECT_NAME}_fit ${PROJECT_NAME}_common)
target_include_directories(${PROJECT_NAME}_fit PUBLIC ${COMMON_INCLUDE_DIRECTORYS} .)
set_target_properties(${PROJECT_NAME}_fit PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME}_fit REUSE_FROM ${PROJECT_NAME}_common)
target_compile_definitions(${PROJECT_NAME}_fit PRIVATE "_XOPEN_SOURCE")

install(TARGETS ${PROJECT_NAME}_fit RUNTIME DESTINATION bin)
