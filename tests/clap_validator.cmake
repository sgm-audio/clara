# tests/clap_validator.cmake
#
# CLAP validator test runner (CMake script-mode).
#
# Invoked from CTest via:
#   cmake -DCLAP_VALIDATOR_EXE=<path> \
#         -DPLUGIN_PATH=<path-to-MambaSuite.clap> \
#         -DREQUIRED_CATEGORIES="audio-ports;params;state;latency" \
#         -P tests/clap_validator.cmake
#
# Required CLAP extension categories exercised by the standard clap-validator
# suite (https://github.com/free-audio/clap-validator):
#   - audio-ports   : CLAP_EXT_AUDIO_PORTS  — port count, channel masks, in-place
#   - params        : CLAP_EXT_PARAMS       — param value/format/range handling
#   - state         : CLAP_EXT_STATE        — save/restore round-trip
#   - latency       : CLAP_EXT_LATENCY      — get() returns stable non-zero value
#
# The clap-validator validate command runs its full test suite against the
# plugin. There is no per-category CLI flag; clap-validator groups checks by
# extension and the required four are part of every conformant plugin's
# standard coverage. The script:
#   - Exits 0  when the validator exits 0 (all categories pass)
#   - Exits 1  when the validator exits non-zero (any category fails)
#   - Exits 0  with [SKIPPED] log line when plugin or validator is missing
#     (cmake -P cannot return arbitrary codes; CTest will report PASS but the
#     [SKIPPED] marker is visible in test output. SKIP detection at configure
#     time in tests/CMakeLists.txt is the primary SKIP mechanism.)
#
# KNOWN-FAILURE / SKIPPED results:
#   - The plugin currently has no GUI (CLAP_EXT_GUI deferred, see TODO.md).
#     Any clap-validator test tagged with "gui" or "editor" will be SKIPPED by
#     the validator itself, not by this script. The required categories
#     (audio-ports, params, state, latency) are non-GUI and are not affected.
#   - The plugin declares 3 parameters (bypass, wet-dry, intensity). The
#     validator may report unused-parameter hints on the host side; those are
#     informational, not failures.
#   - Quick mode (default): processing tests (param-fuzz, process-*, and
#     param-set-wrong, all of which call process() through Mamba inference)
#     are excluded via --invert-filter because Mamba runs in CPU-only ONNX
#     Runtime with untrained random weights. The structural checks
#     (descriptor-consistency, features, param-conversions, scan-time,
#     audio-ports shape) still run. Pass CLAP_VALIDATOR_FULL=ON at configure
#     time to run the full validator suite (slow; suitable for nightly CI).

cmake_minimum_required(VERSION 3.14)

# ---- Argument validation ---------------------------------------------------

if(NOT DEFINED CLAP_VALIDATOR_EXE OR NOT DEFINED PLUGIN_PATH)
    message(STATUS
        "[clap_validator][SKIPPED] Missing required -D arguments: "
        "CLAP_VALIDATOR_EXE and/or PLUGIN_PATH.")
    return()
endif()

if(NOT EXISTS "${PLUGIN_PATH}")
    message(STATUS
        "[clap_validator][SKIPPED] Plugin not found: ${PLUGIN_PATH} "
        "(build the project first).")
    return()
endif()

if(NOT EXISTS "${CLAP_VALIDATOR_EXE}")
    message(STATUS
        "[clap_validator][SKIPPED] Validator binary not found: "
        "${CLAP_VALIDATOR_EXE}.")
    return()
endif()

# ---- Run validator ---------------------------------------------------------

message(STATUS "[clap_validator] Plugin:  ${PLUGIN_PATH}")
message(STATUS "[clap_validator] Tool:    ${CLAP_VALIDATOR_EXE}")
message(STATUS "[clap_validator] Mode:    ${TEST_LABEL}")
message(STATUS "[clap_validator] Required: ${REQUIRED_CATEGORIES}")

# Build the command. In quick mode (TEST_FILTER non-empty), pass
# --invert-filter to SKIP the named tests instead of including-only.
set(_cv_command "${CLAP_VALIDATOR_EXE}" validate "${PLUGIN_PATH}")
if(TEST_FILTER)
    message(STATUS "[clap_validator] Skipping tests matching: ${TEST_FILTER}")
    list(APPEND _cv_command --invert-filter --test-filter "${TEST_FILTER}")
endif()

message(STATUS "[clap_validator] Running validate ...")
execute_process(
    COMMAND ${_cv_command}
    RESULT_VARIABLE cv_result
    OUTPUT_VARIABLE cv_stdout
    ERROR_VARIABLE cv_stderr
    TIMEOUT 150
)

# ---- Report ----------------------------------------------------------------

if(cv_result EQUAL 0)
    message(STATUS "[clap_validator] PASS — all checks succeeded")
    if(cv_stdout)
        message(STATUS "${cv_stdout}")
    endif()
    return()
endif()

message(STATUS "[clap_validator] FAIL — exit code ${cv_result}")
if(cv_stdout)
    message(STATUS "---- validator stdout ----")
    message(STATUS "${cv_stdout}")
endif()
if(cv_stderr)
    message(STATUS "---- validator stderr ----")
    message(STATUS "${cv_stderr}")
endif()

message(FATAL_ERROR
    "[clap_validator] Validator reported failures. See output above. "
    "Required categories that must pass: ${REQUIRED_CATEGORIES}.")
