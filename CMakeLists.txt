cmake_minimum_required(VERSION 3.14)
project(MambaAudioSuite VERSION 0.1.0 LANGUAGES C CXX)

# CTest must be enabled at the top level (before any add_subdirectory) for
# `enable_testing()` to register tests from subdirectories. Guarded by
# BUILD_TESTING so configure-time ctest configuration is only performed when
# the user opts in (cmake -DBUILD_TESTING=ON).
if(BUILD_TESTING)
    enable_testing()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /FS /EHsc")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /fp:fast /DNDEBUG /GR- /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /DDEBUG /MDd /FS /EHsc")
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror=return-type")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -DNDEBUG -fno-rtti")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")  # ponytail: POSITION_INDEPENDENT_CODE doesn't propagate to C for MODULE targets on CMake 4.3
endif()

if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_BUILD_RPATH "@loader_path")
elseif(WIN32)
    # DLL search path: no RPATH needed on Windows
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_RPATH "$ORIGIN")
endif()

include(FetchContent)

# 1. Fetch CLAP (pinned to 1.2.7)
FetchContent_Declare(
    clap
    GIT_REPOSITORY https://github.com/free-audio/clap.git
    GIT_TAG 29ffcc273be7c7c651f6c9953b99e69700e2387a
)
FetchContent_MakeAvailable(clap)

# 2. Fetch ARA SDK (pinned)
FetchContent_Declare(
    ara_sdk
    GIT_REPOSITORY https://github.com/Celemony/ARA_SDK.git
    GIT_TAG releases/2.3.0
)
FetchContent_MakeAvailable(ara_sdk)

# 3. Fetch clap-wrapper (for VST3 + AU builds)
#
# Optional: the VST3 target is the Ableton path but pulls in clap-wrapper, CPM and
# the VST3 SDK. Set -DCLARA_BUILD_VST3=OFF to build the CLAP plugin, ARA library
# and tests without any of that (useful on restricted networks / minimal CI).
option(CLARA_BUILD_VST3 "Build the VST3 wrapper via clap-wrapper (fetches VST3 SDK)" ON)

# We only declare it here; add_subdirectory is used later for the VST3 target.
set(CLAP_SDK_ROOT ${clap_SOURCE_DIR} CACHE INTERNAL "CLAP SDK root for clap-wrapper")
set(CLAP_WRAPPER_DOWNLOAD_DEPENDENCIES ON CACHE INTERNAL "Auto-download VST3 SDK")

# clap-wrapper downloads the VST3 SDK through CPM, whose bootstrap fetches
# CPM.cmake from a GitHub *release asset*. That fails in environments where git
# works but release-asset downloads are blocked. clap-wrapper skips CPM entirely
# when VST3_SDK_ROOT is set, so fetch the same SDK revision it would have fetched,
# via git, and point it there. Set -DVST3_SDK_ROOT=/path/to/sdk to use your own.
if(CLARA_BUILD_VST3 AND VST3_SDK_ROOT STREQUAL "")
    FetchContent_Declare(
        clara_vst3sdk
        GIT_REPOSITORY https://github.com/steinbergmedia/vst3sdk.git
        GIT_TAG v3.7.6_build_18
        GIT_SUBMODULES base public.sdk pluginterfaces
        GIT_SHALLOW FALSE
    )
    FetchContent_GetProperties(clara_vst3sdk)
    if(NOT clara_vst3sdk_POPULATED)
        message(STATUS "CLARA: fetching VST3 SDK v3.7.6_build_18 via git (bypasses clap-wrapper's CPM bootstrap)")
        FetchContent_Populate(clara_vst3sdk)
    endif()
    if(EXISTS "${clara_vst3sdk_SOURCE_DIR}/public.sdk")
        set(VST3_SDK_ROOT "${clara_vst3sdk_SOURCE_DIR}" CACHE PATH "VST3 SDK root for clap-wrapper" FORCE)
    else()
        message(WARNING "CLARA: VST3 SDK fetch incomplete; clap-wrapper will fall back to CPM")
    endif()
endif()

if(CLARA_BUILD_VST3)
    FetchContent_Declare(
        clap_wrapper
        GIT_REPOSITORY https://github.com/free-audio/clap-wrapper.git
        GIT_TAG v0.9.1
    )
    # Populate without MakeAvailable to avoid running its CMakeLists.txt twice
    FetchContent_GetProperties(clap_wrapper)
    if(NOT clap_wrapper_POPULATED)
        FetchContent_Populate(clap_wrapper)
    endif()

    # clap-wrapper's bootstrap stub calls file(DOWNLOAD ... EXPECTED_HASH) on
    # every configure, so seeding the cache file above is not enough: if the
    # download host is unreachable the configure aborts even though the file is
    # already present. Wrap that call in an existence guard. This is idempotent
    # (the guard is only inserted when the unguarded pattern is found) and is a
    # no-op on any clap-wrapper revision that already guards it.
    set(_clara_cpm_stub "${clap_wrapper_SOURCE_DIR}/cmake/external/CPM.cmake")
    if(EXISTS "${_clara_cpm_stub}")
        file(READ "${_clara_cpm_stub}" _clara_stub_src)
        if(NOT _clara_stub_src MATCHES "if\(NOT EXISTS \$\{CPM_DOWNLOAD_LOCATION\}\)")
            string(REPLACE
                "file(DOWNLOAD\n     https://github.com/cpm-cmake/CPM.cmake/releases/download/v\${CPM_DOWNLOAD_VERSION}/CPM.cmake\n     \${CPM_DOWNLOAD_LOCATION} EXPECTED_HASH SHA256=\${CPM_HASH_SUM}\n)"
                "if(NOT EXISTS \${CPM_DOWNLOAD_LOCATION})\n  file(DOWNLOAD\n     https://github.com/cpm-cmake/CPM.cmake/releases/download/v\${CPM_DOWNLOAD_VERSION}/CPM.cmake\n     \${CPM_DOWNLOAD_LOCATION} EXPECTED_HASH SHA256=\${CPM_HASH_SUM}\n  )\nendif()"
                _clara_stub_patched "${_clara_stub_src}")
            if(NOT _clara_stub_patched STREQUAL _clara_stub_src)
                file(WRITE "${_clara_cpm_stub}" "${_clara_stub_patched}")
                message(STATUS "CLARA: guarded clap-wrapper's CPM download (offline-tolerant bootstrap)")
            else()
                message(STATUS "CLARA: clap-wrapper CPM stub has an unexpected shape; leaving it alone")
            endif()
        endif()
    endif()
endif()

# Common include directories
set(COMMON_INCLUDES
    ${clap_SOURCE_DIR}/include
    ${ara_sdk_SOURCE_DIR}
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Dispatch
)

# --- Mamba-3 Core (custom C++ engine, flat binary weights) ---
add_library(Mamba3Core STATIC
    src/core/SsmBlock.cpp
    src/core/SsmInferenceCore.cpp
    src/core/WeightLoader.cpp
    src/core/MambaConfig.cpp
)
target_include_directories(Mamba3Core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(Mamba3Core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)
# SPSCQueue.hpp is header-only — included via MambaClapPlugin.hpp, no compilation unit needed

# --- CLAP Plugin ---
add_library(MambaSuiteClap MODULE src/clap/MambaClapPlugin.cpp)
target_include_directories(MambaSuiteClap PRIVATE ${COMMON_INCLUDES})
set_target_properties(MambaSuiteClap PROPERTIES
    PREFIX ""
    SUFFIX ".clap"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)
if(WIN32)
    # DLL search path: no RPATH needed
elseif(APPLE)
    set_property(TARGET MambaSuiteClap PROPERTY INSTALL_RPATH "@loader_path")
    set_property(TARGET MambaSuiteClap PROPERTY BUILD_RPATH "@loader_path")
    set_property(TARGET MambaSuiteClap PROPERTY BUILD_WITH_INSTALL_RPATH ON)
else()
    set_property(TARGET MambaSuiteClap PROPERTY INSTALL_RPATH "$ORIGIN")
    set_property(TARGET MambaSuiteClap PROPERTY BUILD_RPATH "$ORIGIN")
    set_property(TARGET MambaSuiteClap PROPERTY BUILD_WITH_INSTALL_RPATH ON)
endif()
target_link_libraries(MambaSuiteClap
    PRIVATE
    Mamba3Core
)
if(UNIX AND NOT APPLE)
    target_link_libraries(MambaSuiteClap PRIVATE pthread dl X11)
endif()

# --- ARA Plugin ---
set(ARA_SOURCES
    src/ara/MambaAraController.cpp
    ${ara_sdk_SOURCE_DIR}/ARA_Library/PlugIn/ARAPlug.cpp
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Debug/ARADebug.c
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Utilities/ARAChannelFormat.cpp
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Utilities/ARAPitchInterpretation.cpp
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Dispatch/ARAPlugInDispatch.cpp
)

add_library(MambaSuiteARA MODULE ${ARA_SOURCES})
target_include_directories(MambaSuiteARA PRIVATE ${COMMON_INCLUDES}
    ${ara_sdk_SOURCE_DIR}/ARA_Library
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Debug
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Dispatch
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Utilities
    ${ara_sdk_SOURCE_DIR}/ARA_API
)
if(WIN32)
    set(ARA_SUFFIX ".dll")
elseif(APPLE)
    set(ARA_SUFFIX ".dylib")
else()
    set(ARA_SUFFIX ".so")
endif()
set_target_properties(MambaSuiteARA PROPERTIES
    PREFIX ""
    SUFFIX "${ARA_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)
if(WIN32)
    # DLL search path: no RPATH needed
elseif(APPLE)
    set_property(TARGET MambaSuiteARA PROPERTY INSTALL_RPATH "@loader_path")
    set_property(TARGET MambaSuiteARA PROPERTY BUILD_RPATH "@loader_path")
    set_property(TARGET MambaSuiteARA PROPERTY BUILD_WITH_INSTALL_RPATH ON)
else()
    set_property(TARGET MambaSuiteARA PROPERTY INSTALL_RPATH "$ORIGIN")
    set_property(TARGET MambaSuiteARA PROPERTY BUILD_RPATH "$ORIGIN")
    set_property(TARGET MambaSuiteARA PROPERTY BUILD_WITH_INSTALL_RPATH ON)
endif()
target_link_libraries(MambaSuiteARA
    PRIVATE
    Mamba3Core
)
if(UNIX AND NOT APPLE)
    target_link_libraries(MambaSuiteARA PRIVATE pthread dl)
endif()

# --- VST3 Wrapper (via clap-wrapper) ---
# Wraps the built MambaSuiteClap.clap into a native VST3 for Ableton Live etc.
set(CLAP_WRAPPER_OUTPUT_NAME "MambaSuite" CACHE STRING "VST3 output name")
set(CLAP_WRAPPER_BUNDLE_IDENTIFIER "com.clara.mambasuite" CACHE STRING "macOS bundle ID")
set(CLAP_WRAPPER_BUNDLE_VERSION "${PROJECT_VERSION}" CACHE STRING "Bundle version")
set(CLAP_WRAPPER_BUILD_STANDALONE OFF CACHE BOOL "Skip standalone target")
set(CLAP_WRAPPER_BUILD_AUV2 OFF CACHE BOOL "Skip AUv2 target for now")

# clap-wrapper creates MambaSuite_as_vst3 in its own subdirectory scope.
# Suppress GCC 16 warnings in clap-wrapper's bundled fmt library (older fmt + GCC 16).
# GCC-only flags: MSVC/Clang reject or ignore them, and this variable is
# directory-scoped so it reaches every target, not just clap-wrapper.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-dangling-reference -Wno-tautological-compare")
endif()
if(CLARA_BUILD_VST3)
    add_subdirectory(${clap_wrapper_SOURCE_DIR} build/clap_wrapper_build EXCLUDE_FROM_ALL)
endif()

# GCC 16 false positive in clap-wrapper's vendored sha1.cpp (stringop-overflow
# under inlining at -O3); clap-wrapper adds -Werror unconditionally, so demote
# just this diagnostic on its interface target.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND TARGET clap-wrapper-compile-options)
    target_compile_options(clap-wrapper-compile-options INTERFACE -Wno-error=stringop-overflow)
endif()

# Strip release builds (Unix linker flag — generator-expr for multi-config support)
target_link_options(MambaSuiteClap PRIVATE $<$<AND:$<PLATFORM_ID:Linux,Darwin>,$<CONFIG:Release>>:-s>)
target_link_options(MambaSuiteARA PRIVATE $<$<AND:$<PLATFORM_ID:Linux,Darwin>,$<CONFIG:Release>>:-s>)

# --- VST3 target configuration (only when the wrapper is enabled) ---
if(CLARA_BUILD_VST3)
    # Set output directory and model deploy via global properties
    set_property(TARGET MambaSuite_as_vst3 PROPERTY
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins)
    set_property(TARGET MambaSuite_as_vst3 PROPERTY
        CXX_VISIBILITY_PRESET hidden)
    set_property(TARGET MambaSuite_as_vst3 PROPERTY
        POSITION_INDEPENDENT_CODE ON)

    target_link_options(MambaSuite_as_vst3 PRIVATE $<$<AND:$<PLATFORM_ID:Linux,Darwin>,$<CONFIG:Release>>:-s>)

    # Deploy models via install rule (works across directory boundaries)
    if(EXISTS ${CMAKE_SOURCE_DIR}/models)
        install(DIRECTORY ${CMAKE_SOURCE_DIR}/models/
            DESTINATION ${CMAKE_BINARY_DIR}/plugins/MambaSuite.vst3/Contents/Resources/models
            COMPONENT vst3
            FILES_MATCHING PATTERN "*.bin"
        )
    endif()
endif()

# ---------------------------------------------------------------------------
# Per-target post-build deploy (multi-config safe)
#
# Why this is needed:
#   With multi-config generators (Visual Studio, Xcode, Ninja Multi-Config),
#   the plugin binary lands at ${CMAKE_BINARY_DIR}/plugins/<CONFIG>/. The
#   previous unconditional file(COPY ...) blocks deployed models to
#   ${CMAKE_BINARY_DIR}/plugins/ only. mambaResolveModulePath() then computed
#   <binary-dir>/models/model.bin, the file was missing, mamba_init() returned
#   false, and every clap-validator init()-dependent test failed.
#
# POST_BUILD + generator expressions deploy to <TARGET_FILE_DIR>/ for
# every configuration, so single-config (Ninja, Unix Makefiles) and
# multi-config (VS, Xcode) generators both get the right layout.
# ---------------------------------------------------------------------------

# Models -> <binary-dir>/models/
if(EXISTS ${CMAKE_SOURCE_DIR}/models)
    add_custom_command(TARGET MambaSuiteClap POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/models
            $<TARGET_FILE_DIR:MambaSuiteClap>/models
        COMMENT "Deploying models to $<TARGET_FILE_DIR:MambaSuiteClap>/models"
        VERBATIM
    )
    add_custom_command(TARGET MambaSuiteARA POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/models
            $<TARGET_FILE_DIR:MambaSuiteARA>/models
        COMMENT "Deploying models to $<TARGET_FILE_DIR:MambaSuiteARA>/models"
        VERBATIM
    )
endif()



# ---------------------------------------------------------------------------
# --- Unit Tests (opt-in: cmake -DBUILD_TESTING=ON) ---
add_subdirectory(tests)

# ---------------------------------------------------------------------------
# --- `clara_test` custom target (runs ctest) ---
# Convenience: `cmake --build . --target clara_test` builds the plugins and
# runs the CTest suite. This ensures the .clap binary exists before the
# clap_validator test is executed.
#
# The target name cannot be `test` — that is reserved by CTest when testing
# is enabled.
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
    set(_clara_test_deps MambaSuiteClap MambaSuiteARA)
    if(CLARA_BUILD_VST3)
        list(APPEND _clara_test_deps MambaSuite_as_vst3)
    endif()
    add_custom_target(clara_test
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS ${_clara_test_deps}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running CTest (unit tests + clap-validator)"
    )
endif()

# ---------------------------------------------------------------------------
# Install Rules
# ---------------------------------------------------------------------------
#
# CLAP bundle structure (for hosts that discover bundles):
#   ${CMAKE_INSTALL_PREFIX}/lib/clap/MambaSuiteClap.clap/
#     Contents/
#       linux-x86-64/
#         MambaSuiteClap.clap    (the plugin binary)
#         libmamba_core.so
#         models/
#           model.bin
#
# ARA standalone:
#   ${CMAKE_INSTALL_PREFIX}/lib/ara/MambaSuiteARA.so
#
# Bare CLAP binary (for hosts that scan bare .clap files):
#   ${CMAKE_INSTALL_PREFIX}/lib/clap/MambaSuiteClap.clap
#
# An installed plugin directory includes everything needed at runtime
# (plugin + shared engine + model). The RUNPATH is $ORIGIN so all
# dependencies resolve relative to the plugin binary location.
# ---------------------------------------------------------------------------

# CLAP bundle install destinations per CLAP specification:
#   Windows: %COMMONPROGRAMFILES%\CLAP\<name>.clap\
#   macOS:   /Library/Audio/Plug-Ins/CLAP/<name>.clap/
#   Linux:   /usr/lib/clap/<name>.clap/
# Set CMAKE_INSTALL_PREFIX to override (e.g., $HOME/.clap for per-user install).
if(WIN32)
    set(CLAP_INSTALL_PREFIX "CLAP" CACHE STRING "CLAP install directory relative to CMAKE_INSTALL_PREFIX")
    set(CLAP_BUNDLE_DIR "${CLAP_INSTALL_PREFIX}/MambaSuiteClap.clap/Contents/x86_64-windows")
elseif(APPLE)
    set(CLAP_INSTALL_PREFIX "lib/clap" CACHE STRING "CLAP install directory relative to CMAKE_INSTALL_PREFIX")
    set(CLAP_BUNDLE_DIR "${CLAP_INSTALL_PREFIX}/MambaSuiteClap.clap/Contents/MacOS")
else()
    set(CLAP_INSTALL_PREFIX "lib/clap" CACHE STRING "CLAP install directory relative to CMAKE_INSTALL_PREFIX")
    set(CLAP_BUNDLE_DIR "${CLAP_INSTALL_PREFIX}/MambaSuiteClap.clap/Contents/linux-x86-64")
endif()

# --- CLAP bundle install ---
install(TARGETS MambaSuiteClap
    LIBRARY DESTINATION ${CLAP_BUNDLE_DIR}
    COMPONENT clap
)

# Also install bare .clap for hosts that don't support bundles
install(FILES ${CMAKE_BINARY_DIR}/plugins/MambaSuiteClap.clap
    DESTINATION ${CLAP_INSTALL_PREFIX}
    COMPONENT clap
)

# --- ARA install ---
set(ARA_INSTALL_PREFIX "lib/ara" CACHE STRING "ARA install directory relative to CMAKE_INSTALL_PREFIX")
install(TARGETS MambaSuiteARA
    LIBRARY DESTINATION ${ARA_INSTALL_PREFIX}
    COMPONENT ara
    RUNTIME DESTINATION ${ARA_INSTALL_PREFIX}
)

# --- Models for CLAP bundle ---
install(DIRECTORY ${CMAKE_SOURCE_DIR}/models/
    DESTINATION ${CLAP_BUNDLE_DIR}/models
    COMPONENT clap
    FILES_MATCHING PATTERN "*.bin"
)

# --- Models for ARA ---
install(DIRECTORY ${CMAKE_SOURCE_DIR}/models/
    DESTINATION ${ARA_INSTALL_PREFIX}/models
    COMPONENT ara
    FILES_MATCHING PATTERN "*.bin"
)

# Convenience: `cmake --install . --component clap` for CLAP only
#              `cmake --install . --component ara` for ARA only
#              `cmake --install .` for both
