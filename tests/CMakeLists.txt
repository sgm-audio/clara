# Unit tests for CLARA Mamba Audio Suite
#
# Build with: cmake -B build -DBUILD_TESTING=ON
# Run with:   ctest --test-dir build --output-on-failure

if(NOT BUILD_TESTING)
    return()
endif()

enable_testing()

# --- Test: MambaInferenceEngine smoke test ---
add_executable(test_engine
    test_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/core/MambaInferenceEngine.cpp
)
target_include_directories(test_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(test_engine PRIVATE ${ONNXRUNTIME_LIB} pthread dl)

# Pass model path so test can locate mamba_rt.onnx
target_compile_definitions(test_engine PRIVATE
    TEST_MODEL_DIR="${CMAKE_SOURCE_DIR}/models"
)

add_test(NAME test_engine COMMAND test_engine)
