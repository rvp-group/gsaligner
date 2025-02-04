enable_testing()

add_executable(test_photometric test_photometric.cu)
target_link_libraries(test_photometric PRIVATE GTest::gtest_main tools registration ${OpenCV_LIBS})

include(GoogleTest)
gtest_discover_tests(test_photometric)

# add_custom_target(run_tests
#     COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
#     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests/  # Change to the build directory
#     DEPENDS test_algebra_operations test_cuda_matrix test_projections
#             test_hash_utils test_semantic_hash_utils test_streamer
#             test_grid_serializer
# )
# message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
