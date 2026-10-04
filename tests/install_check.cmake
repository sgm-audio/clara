# tests/install_check.cmake
#
# Install-layout regression test (CMake script mode).
#
# Invoked from CTest as:
#   ctest --test-dir build -R test_install_layout
#
# What it guards (all of this was broken before 2026-10-04):
#   1. Every install component exits 0. `--component clap` used to fail because
#      a bare `MambaSuiteClap.clap` file was installed into the same path as the
#      `MambaSuiteClap.clap/` bundle directory, self-nesting the tree and making
#      the subsequent models install die with "Permission denied".
#   2. The install honours --prefix. `--component vst3` used an absolute
#      DESTINATION pointing into the build tree, so --prefix was ignored — and
#      the VST3 module itself was never installed at all, only its weights.
#   3. The layout matches what the plugin needs at runtime: the .clap sits in a
#      directory that also contains models/model.bin, because
#      mambaResolveModulePath() resolves weights next to the loaded .clap.
#   4. The CLAP output stem matches the VST3 wrapper's stem. clap-wrapper looks
#      for "<vst3-stem>.clap"; if the two drift, the VST3 loads nothing at all.
#
# Multi-config generators (Visual Studio, Xcode, Ninja Multi-Config) put the
# built plugin at <build>/plugins/<CONFIG>/, so both locations are searched.
#
# SKIP semantics: exits 77 when the build tree is not configured (the script is
# checked in, so a stray invocation must not look like a failure).

cmake_minimum_required(VERSION 3.14)

if(NOT DEFINED BUILD_DIR OR NOT DEFINED CLAP_OUTPUT_NAME)
    message(STATUS "[install_check][SKIPPED] Missing -D arguments (BUILD_DIR, CLAP_OUTPUT_NAME)")
    return()
endif()

# ---- Locate the built plugins (single-config and multi-config layouts) -----
file(GLOB _built_claps "${BUILD_DIR}/plugins/*.clap" "${BUILD_DIR}/plugins/*/*.clap")
set(_built_clap_files "")
foreach(_entry IN LISTS _built_claps)
    if(NOT IS_DIRECTORY "${_entry}")
        list(APPEND _built_clap_files "${_entry}")
    endif()
endforeach()
if(NOT _built_clap_files)
    message(STATUS "[install_check][SKIPPED] no built .clap under ${BUILD_DIR}/plugins (build the project first).")
    return()
endif()

# The expected file must exist. Extra .clap files are tolerated (an incremental
# build tree can hold an artefact from before a rename) but a tree that holds
# only differently-named plugins is the drift this test exists to catch.
set(_expected_built FALSE)
foreach(_entry IN LISTS _built_clap_files)
    get_filename_component(_stem "${_entry}" NAME_WE)
    if(_stem STREQUAL CLAP_OUTPUT_NAME)
        set(_expected_built TRUE)
    endif()
endforeach()
if(NOT _expected_built)
    message(FATAL_ERROR
        "No '${CLAP_OUTPUT_NAME}.clap' under ${BUILD_DIR}/plugins, found: ${_built_clap_files}\n"
        "CLAP_WRAPPER_OUTPUT_NAME must name both the CLAP file and the VST3 wrapper: the wrapper\n"
        "searches the CLAP paths for '<vst3-stem>.clap' (clap-wrapper wrapasvst3_entry.cpp:\n"
        "plugname = getBinaryName() + \".clap\"), so a 'MambaSuiteClap.clap' + 'MambaSuite.vst3'\n"
        "pair loads nothing. Rename the CLAP output (or the wrapper) so the stems match.")
endif()

file(GLOB _built_vst3_dirs "${BUILD_DIR}/*/*.vst3" "${BUILD_DIR}/*.vst3")
list(FILTER _built_vst3_dirs EXCLUDE REGEX "/plugins/")

# ---- 1+2. Each component installs, honouring --prefix -----------------------
set(_prefix "${BUILD_DIR}/install_check_prefix")
file(REMOVE_RECURSE "${_prefix}")

foreach(_component clap ara)
    execute_process(
        COMMAND ${CMAKE_COMMAND} --install "${BUILD_DIR}" --component ${_component} --prefix "${_prefix}"
        RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "install --component ${_component} failed (exit ${_rc})\n${_out}\n${_err}")
    endif()
    message(STATUS "[install_check] --component ${_component}: exit 0")
endforeach()

# The vst3 component only carries anything when the wrapper was built. A missing
# component is a silent no-op in CMake (exit 0, nothing installed), so check the
# build tree to decide whether its absence is expected.
if(_built_vst3_dirs)
    execute_process(
        COMMAND ${CMAKE_COMMAND} --install "${BUILD_DIR}" --component vst3 --prefix "${_prefix}"
        RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "install --component vst3 failed (exit ${_rc})\n${_out}\n${_err}")
    endif()
    message(STATUS "[install_check] --component vst3: exit 0")
else()
    message(STATUS "[install_check] no built .vst3 found (CLARA_BUILD_VST3=OFF) — vst3 component not checked")
endif()

# ---- 3. Layout: weights beside the plugin binary ---------------------------
# Linux installs the CLAP as a bare file; macOS/Windows install a bundle whose
# arch directory holds the binary. Accept either, then require models/model.bin
# in the directory that contains the plugin binary.
file(GLOB_RECURSE _installed_claps "${_prefix}/*.clap")
set(_installed_files "")
foreach(_entry IN LISTS _installed_claps)
    if(IS_DIRECTORY "${_entry}")
        continue()
    endif()
    list(APPEND _installed_files "${_entry}")
endforeach()
set(_installed_claps "${_installed_files}")

if(NOT _installed_claps)
    message(FATAL_ERROR "No installed .clap under ${_prefix} — is the clap install rule wired up?")
endif()
list(LENGTH _installed_claps _clap_count)
if(NOT _clap_count EQUAL 1)
    message(FATAL_ERROR "Expected exactly one installed .clap, found ${_clap_count}: ${_installed_claps}")
endif()

get_filename_component(_installed_dir "${_installed_claps}" DIRECTORY)
get_filename_component(_installed_name "${_installed_claps}" NAME_WE)

# ---- 4. The name clap-wrapper will search for ------------------------------
if(NOT _installed_name STREQUAL CLAP_OUTPUT_NAME)
    message(FATAL_ERROR
        "Installed .clap is named '${_installed_name}' but the VST3 wrapper searches for "
        "'${CLAP_OUTPUT_NAME}.clap' — the stems must match (clap-wrapper wrapasvst3_entry.cpp).")
endif()

if(NOT EXISTS "${_installed_dir}/models/model.bin")
    message(FATAL_ERROR
        "models/model.bin is not installed next to ${_installed_name}.clap (${_installed_dir}); "
        "the plugin resolves its weights with dladdr relative to the loaded .clap.")
endif()

# ---- 5. The VST3 actually installed ----------------------------------------
if(_built_vst3_dirs)
    file(GLOB_RECURSE _installed_vst3 "${_prefix}/*/*.so" "${_prefix}/*/*.dll" "${_prefix}/*/*.dylib")
    set(_vst3_modules "")
    foreach(_entry IN LISTS _installed_vst3)
        if(_entry MATCHES "\\.vst3/Contents/")
            list(APPEND _vst3_modules "${_entry}")
        endif()
    endforeach()
    if(NOT _vst3_modules)
        message(FATAL_ERROR
            "A VST3 was built (${_built_vst3_dirs}) but --component vst3 installed no module under "
            "${_prefix}. The install rule must ship the wrapper binary, not just the weights.")
    endif()
    list(GET _vst3_modules 0 _vst3_module)
    get_filename_component(_vst3_stem "${_vst3_module}" NAME_WE)
    if(NOT _vst3_stem STREQUAL CLAP_OUTPUT_NAME)
        message(FATAL_ERROR
            "Installed VST3 module is '${_vst3_stem}' but the CLAP is '${CLAP_OUTPUT_NAME}' — "
            "clap-wrapper matches them by name and will load nothing.")
    endif()
    message(STATUS "[install_check] VST3 module: ${_vst3_module} (stem matches the CLAP)")
endif()

message(STATUS "[install_check] layout OK: ${_installed_claps} + models/model.bin")
message(STATUS "[install_check] PASS")
