# Unit tests for CLARA Mamba Audio Suite
#
# Build with: cmake -B build -DBUILD_TESTING=ON
# Run with:   ctest --test-dir build --output-on-failure

if(NOT BUILD_TESTING)
    return()
endif()

# NOTE: enable_testing() is called in the top-level CMakeLists.txt (it must
# be at the top level, before any add_subdirectory(), per CMake docs).

find_package(Threads REQUIRED)

# --- Generate Mamba-3 model.bin for tests ---
# ponytail: runs Python export script once, produces models/model.bin
# Only runs if the output doesn't exist yet.
set(MAMBA3_MODEL_BIN "${CMAKE_SOURCE_DIR}/models/model.bin")
if(NOT EXISTS "${MAMBA3_MODEL_BIN}")
    find_program(PYTHON3_EXE python3)
    if(PYTHON3_EXE)
        execute_process(
            COMMAND ${PYTHON3_EXE}
                "${CMAKE_SOURCE_DIR}/scripts/export_mamba_weights.py"
                --output "${MAMBA3_MODEL_BIN}" --seed 42
            RESULT_VARIABLE _m3_result
            OUTPUT_VARIABLE _m3_out
            ERROR_VARIABLE  _m3_err
            TIMEOUT 60
        )
        if(NOT _m3_result EQUAL 0)
            message(WARNING "Failed to generate ${MAMBA3_MODEL_BIN}: ${_m3_err}")
        endif()
    else()
        message(WARNING "python3 not found — Mamba-3 tests will SKIP. Install python3 to enable.")
    endif()
endif()

# --- Test: Mamba-3 Inference Core ---
add_executable(test_mamba3 test_mamba3.cpp)
target_include_directories(test_mamba3 PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_mamba3 PRIVATE Mamba3Core)
target_compile_definitions(test_mamba3 PRIVATE
    TEST_MODEL_DIR="${CMAKE_SOURCE_DIR}/models"
)
add_test(NAME test_mamba3 COMMAND test_mamba3 "${CMAKE_SOURCE_DIR}/models/model.bin")

# --- Generate Python reference outputs for bit-exact C++ comparison ---
# Runs once, dumps input_ref.bin + output_ref.bin next to model.bin.
# test_mamba3.cpp loads these for per-sample Python↔C++ comparison.
set(MAMBA3_REF_INPUT "${CMAKE_SOURCE_DIR}/models/input_ref.bin")
set(MAMBA3_REF_OUTPUT "${CMAKE_SOURCE_DIR}/models/output_ref.bin")
set(MAMBA3_REF_MODEL "${CMAKE_SOURCE_DIR}/models/model_ref.bin")
if(NOT EXISTS "${MAMBA3_REF_INPUT}" OR NOT EXISTS "${MAMBA3_REF_OUTPUT}" OR NOT EXISTS "${MAMBA3_REF_MODEL}")
    find_program(PYTHON3_EXE python3)
    if(PYTHON3_EXE)
        add_custom_target(generate_reference_outputs ALL
            COMMAND ${PYTHON3_EXE}
                "${CMAKE_SOURCE_DIR}/scripts/dump_reference_output.py"
                --output-dir "${CMAKE_SOURCE_DIR}/models"
                --seed 42
            COMMENT "Generating Python↔C++ reference outputs..."
            BYPRODUCTS
                "${MAMBA3_REF_INPUT}"
                "${MAMBA3_REF_OUTPUT}"
                "${MAMBA3_REF_MODEL}"
        )
        add_dependencies(test_mamba3 generate_reference_outputs)
    endif()
endif()

# --- Test: CLAP plugin (Mamba-3) ---
add_executable(test_clap_plugin
    test_clap_plugin.cpp
    ${CMAKE_SOURCE_DIR}/src/clap/MambaClapPlugin.cpp
)
target_include_directories(test_clap_plugin PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/clap
    ${CMAKE_SOURCE_DIR}/src/core
    ${clap_SOURCE_DIR}/include
)
target_link_libraries(test_clap_plugin PRIVATE Mamba3Core)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_clap_plugin PRIVATE pthread dl)
endif()
target_compile_definitions(test_clap_plugin PRIVATE
    TEST_MODEL_DIR="${CMAKE_SOURCE_DIR}/models"
)
add_test(NAME test_clap_plugin COMMAND test_clap_plugin)

# --- Test: ARA controller (Mamba-3) ---
add_executable(test_ara_controller
    test_ara_controller.cpp
    ${CMAKE_SOURCE_DIR}/src/ara/MambaAraController.cpp
    ${ara_sdk_SOURCE_DIR}/ARA_Library/PlugIn/ARAPlug.cpp
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Debug/ARADebug.c
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Utilities/ARAChannelFormat.cpp
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Utilities/ARAPitchInterpretation.cpp
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Dispatch/ARAPlugInDispatch.cpp
)
target_include_directories(test_ara_controller PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/ara
    ${ara_sdk_SOURCE_DIR}
    ${ara_sdk_SOURCE_DIR}/ARA_Library
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Debug
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Dispatch
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Utilities
    ${ara_sdk_SOURCE_DIR}/ARA_API
)
target_link_libraries(test_ara_controller PRIVATE Mamba3Core)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_ara_controller PRIVATE pthread dl)
endif()

add_test(NAME test_ara_controller COMMAND test_ara_controller)

# ---------------------------------------------------------------------------
# CLAP validator integration test
#
# Runs clap-validator (https://github.com/free-audio/clap-validator) against
# the built MambaSuiteClap.clap. Required categories covered by the standard
# clap-validator suite: audio-ports, params, state, latency.
#
# Resolution order:
#   1. find_program() — local install on PATH
#   2. cargo install — fetch and build from crates.io (if cargo is on PATH)
#   3. Skip the test (do not add it) and inform the user
# ---------------------------------------------------------------------------

set(_clap_validator_required_categories "audio-ports;params;state;latency")

# Quick mode skips processing tests because a full 21-case validation pass
# with process() calls takes many minutes per call and is impractical for
# CI. Quick mode exercises the four required categories at the structural /
# lifecycle layer (init, descriptor, audio-ports shape, param roundtrip, state
# save/restore, latency) without invoking the Mamba forward path.
#
# Set CLAP_VALIDATOR_FULL=ON to run the full validator suite (slow; for
# nightly CI or local debugging, not for every PR).
option(CLAP_VALIDATOR_FULL "Run the full clap-validator suite (slow; includes audio processing tests)" OFF)
if(CLAP_VALIDATOR_FULL)
    set(_clap_validator_test_filter "")
    set(_clap_validator_test_label "full")
else()
    # Invert-filter regex: SKIP any test whose name contains these substrings.
    # The slow offenders all reach the Mamba forward path through process():
    #   - param-fuzz-basic          (50 random param sets, process 5 buffers each)
    #   - param-set-wrong-namespace (calls process to verify param unchanged)
    #   - process-audio-out-of-place-basic / process-note-* (process audio)
    #   - state-reproducibility-*   (roundtrip requires process() per clap docs)
    #   - state-buffered-streams    (same; reproducibility check via process)
    set(_clap_validator_test_filter "fuzz|process|param-set-wrong|state-reproducibility|state-buffered")
    set(_clap_validator_test_label "quick")
endif()

# Use $<TARGET_FILE:MambaSuiteClap> so the plugin path resolves correctly
# for both single-config (Ninja, Unix Makefiles: plugins/MambaSuiteClap.clap)
# and multi-config (Visual Studio, Xcode, Ninja Multi-Config:
# plugins/$<CONFIG>/MambaSuiteClap.clap) generators.

# 1. Local install on PATH
find_program(CLAP_VALIDATOR_EXE
    NAMES clap-validator clap_validator
    DOC   "CLAP plugin validator (https://github.com/free-audio/clap-validator)"
)

# 2. FetchContent / cargo install fallback
#    clap-validator is a Rust project published on GitHub, not crates.io
#    (https://github.com/free-audio/clap-validator). It is installed from
#    master because the last tagged release (0.3.2, 2023) pulls in a version
#    of the `time` crate that does not compile with current Rust toolchains.
if(NOT CLAP_VALIDATOR_EXE)
    find_program(CARGO_EXE cargo DOC "Rust toolchain (cargo)")
    if(CARGO_EXE)
        message(STATUS
            "clap-validator not on PATH; attempting 'cargo install --git "
            "https://github.com/free-audio/clap-validator' (this may take "
            "several minutes on first run) ...")
        execute_process(
            COMMAND "${CARGO_EXE}" install
                --git https://github.com/free-audio/clap-validator.git
                --locked
                clap-validator
            RESULT_VARIABLE _cv_cargo_result
            OUTPUT_VARIABLE _cv_cargo_stdout
            ERROR_VARIABLE _cv_cargo_stderr
            TIMEOUT 1800
        )
        if(_cv_cargo_result EQUAL 0)
            find_program(CLAP_VALIDATOR_EXE
                NAMES clap-validator clap_validator
                HINTS "$ENV{HOME}/.cargo/bin"
                      "$ENV{USERPROFILE}/.cargo/bin"
                NO_DEFAULT_PATH
            )
            if(CLAP_VALIDATOR_EXE)
                message(STATUS "Installed clap-validator: ${CLAP_VALIDATOR_EXE}")
            endif()
        else()
            message(STATUS
                "'cargo install clap-validator' failed (exit ${_cv_cargo_result}). "
                "Validator test will be skipped. stderr: ${_cv_cargo_stderr}")
        endif()
    endif()
endif()

# 3. Register the test (or report SKIP)
if(CLAP_VALIDATOR_EXE)
    add_test(
        NAME clap_validator
        COMMAND ${CMAKE_COMMAND}
            -DCLAP_VALIDATOR_EXE=${CLAP_VALIDATOR_EXE}
            -DPLUGIN_PATH=$<TARGET_FILE:MambaSuiteClap>
            "-DREQUIRED_CATEGORIES=${_clap_validator_required_categories}"
            "-DTEST_FILTER=${_clap_validator_test_filter}"
            "-DTEST_LABEL=${_clap_validator_test_label}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/clap_validator.cmake
    )
    set_tests_properties(clap_validator PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT          180
        LABELS           "clap;validator;integration;${_clap_validator_test_label}"
    )
    message(STATUS "clap-validator test [${_clap_validator_test_label}]: ${CLAP_VALIDATOR_EXE} -> $<TARGET_FILE:MambaSuiteClap>")
else()
    message(STATUS
        "clap-validator not found and cargo not available — "
        "the clap_validator CTest target will be SKIPPED. "
        "Install clap-validator (https://github.com/free-audio/clap-validator) "
        "or Rust + cargo to enable this test.")
endif()
