add_executable(${PROJECT_NAME}_datasetinfo datasetinfo.cpp)
target_link_libraries(${PROJECT_NAME}_datasetinfo ${PROJECT_NAME}_common)
target_include_directories(${PROJECT_NAME}_datasetinfo PUBLIC ${COMMON_INCLUDE_DIRECTORYS} ${SCIPLOT_INCLUDE_DIRECTORYS} .)
set_target_properties(${PROJECT_NAME}_datasetinfo PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS} ${SCIPLOT_LIBRARIES} LINK_FLAGS "")
target_precompile_headers(${PROJECT_NAME}_datasetinfo REUSE_FROM ${PROJECT_NAME}_common)
target_compile_definitions(${PROJECT_NAME}_datasetinfo PRIVATE "_XOPEN_SOURCE")
target_compile_definitions(${PROJECT_NAME}_datasetinfo PRIVATE DRT_ENABLED)

install(TARGETS ${PROJECT_NAME}_datasetinfo RUNTIME DESTINATION bin)
