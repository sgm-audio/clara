cmake_minimum_required(VERSION 3.14)
project(MambaAudioSuite VERSION 1.1.0 LANGUAGES CXX)

# CTest must be enabled at the top level (before any add_subdirectory) for
# `enable_testing()` to register tests from subdirectories. Guarded by
# BUILD_TESTING so configure-time ctest configuration is only performed when
# the user opts in (cmake -DBUILD_TESTING=ON).
if(BUILD_TESTING)
    enable_testing()
endif()

# ponytail: CMake 4.3.3 + GCC 16 workaround — CMAKE_C_COMPILE_OBJECT not
# auto-detected for C language when ARA SDK's ARADebug.c triggers C compilation.
# Remove when CMake >= 4.3.4 or when the C language path is removed.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0" AND NOT DEFINED CMAKE_C_COMPILE_OBJECT)
    set(CMAKE_C_COMPILE_OBJECT "<CMAKE_C_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    # /MD (dynamic CRT): Required because ONNX Runtime's prebuilt onnxruntime.dll
    # is compiled with /MD. Mixing /MT (static CRT) and /MD in the same process
    # causes undefined behavior (separate heaps, incompatible object ownership).
    # Users must install the VC++ Redistributable:
    #   https://aka.ms/vs/17/release/vc_redist.x64.exe
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
    GIT_TAG 1ca0b0aa65a73e932daa4a9402387a8afa5afd17
)
FetchContent_MakeAvailable(ara_sdk)

# 4. Fetch ONNXRuntime (Precompiled binaries)
set(ONNXRUNTIME_VERSION 1.17.1)
if(WIN32)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
elseif(APPLE)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}.tgz")
else()
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
endif()
FetchContent_Declare(
    onnxruntime
    URL ${ONNXRUNTIME_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime)

# ONNXRuntime Setup
set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_LIB_DIR} NO_DEFAULT_PATH)

# Define platform-specific runtime library filenames
if(WIN32)
    set(ONNXRUNTIME_DLL "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll")
elseif(APPLE)
    set(ONNXRUNTIME_DYLIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.dylib")
else()
    set(ONNXRUNTIME_SO "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1.17.1")
endif()

# Common include directories
set(COMMON_INCLUDES
    ${clap_SOURCE_DIR}/include
    ${ara_sdk_SOURCE_DIR}
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Dispatch
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# --- Shared Core (compiled once, linked into both targets) ---
add_library(MambaCore STATIC src/core/MambaInferenceEngine.cpp)
target_include_directories(MambaCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(MambaCore PUBLIC ${ONNXRUNTIME_LIB})
set_target_properties(MambaCore PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

# --- Mamba-3 Core (Phase 3: custom C++ engine, no ONNX) ---
add_library(Mamba3Core STATIC
    src/core/SsmBlock.cpp
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
    MambaCore
)
if(UNIX AND NOT APPLE)
    target_link_libraries(MambaSuiteClap PRIVATE pthread dl)
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
    MambaCore
)
if(UNIX AND NOT APPLE)
    target_link_libraries(MambaSuiteARA PRIVATE pthread dl)
endif()

# Strip release builds (Unix linker flag — generator-expr for multi-config support)
target_link_options(MambaSuiteClap PRIVATE $<$<AND:$<PLATFORM_ID:Linux,Darwin>,$<CONFIG:Release>>:-s>)
target_link_options(MambaSuiteARA PRIVATE $<$<AND:$<PLATFORM_ID:Linux,Darwin>,$<CONFIG:Release>>:-s>)

# ---------------------------------------------------------------------------
# Per-target post-build deploy (multi-config safe)
#
# Why this is needed:
#   With multi-config generators (Visual Studio, Xcode, Ninja Multi-Config),
#   the plugin binary lands at ${CMAKE_BINARY_DIR}/plugins/<CONFIG>/. The
#   previous unconditional file(COPY ...) blocks deployed models and ONNX
#   Runtime to ${CMAKE_BINARY_DIR}/plugins/ only. mambaResolveModulePath()
#   then computed <binary-dir>/models/mamba_rt.onnx, the file was missing,
#   mamba_init() returned false, and every clap-validator init()-dependent
#   test failed.
#
# POST_BUILD + generator expressions deploy to <TARGET_FILE_DIR>/ for
# every configuration, so single-config (Ninja, Unix Makefiles) and
# multi-config (VS, Xcode) generators both get the right layout.
#
# Note: the install() rules at the bottom of this file reference
# ${ONNXRUNTIME_DLL/DYLIB/SO} and ${CMAKE_BINARY_DIR}/plugins/
# libonnxruntime.{so,dylib}. The variable assignments and the
# configure_file() calls below preserve those build-tree paths so the
# install rules continue to work without modification.
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

# ONNX Runtime -> <binary-dir>/
if(WIN32)
    set(ONNXRUNTIME_DLL ${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll)
    if(EXISTS ${ONNXRUNTIME_DLL})
        add_custom_command(TARGET MambaSuiteClap POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_DLL}
                $<TARGET_FILE_DIR:MambaSuiteClap>/onnxruntime.dll
            COMMENT "Deploying onnxruntime.dll to $<TARGET_FILE_DIR:MambaSuiteClap>"
            VERBATIM
        )
        add_custom_command(TARGET MambaSuiteARA POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_DLL}
                $<TARGET_FILE_DIR:MambaSuiteARA>/onnxruntime.dll
            COMMENT "Deploying onnxruntime.dll to $<TARGET_FILE_DIR:MambaSuiteARA>"
            VERBATIM
        )
    endif()
elseif(APPLE)
    set(ONNXRUNTIME_DYLIB ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.${ONNXRUNTIME_VERSION}.dylib)
    if(EXISTS ${ONNXRUNTIME_DYLIB})
        add_custom_command(TARGET MambaSuiteClap POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_DYLIB}
                $<TARGET_FILE_DIR:MambaSuiteClap>/libonnxruntime.${ONNXRUNTIME_VERSION}.dylib
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_DYLIB}
                $<TARGET_FILE_DIR:MambaSuiteClap>/libonnxruntime.dylib
            COMMENT "Deploying libonnxruntime.dylib to $<TARGET_FILE_DIR:MambaSuiteClap>"
            VERBATIM
        )
        add_custom_command(TARGET MambaSuiteARA POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_DYLIB}
                $<TARGET_FILE_DIR:MambaSuiteARA>/libonnxruntime.${ONNXRUNTIME_VERSION}.dylib
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_DYLIB}
                $<TARGET_FILE_DIR:MambaSuiteARA>/libonnxruntime.dylib
            COMMENT "Deploying libonnxruntime.dylib to $<TARGET_FILE_DIR:MambaSuiteARA>"
            VERBATIM
        )
        # Unversioned symlink-named copy in top-level build/plugins/
        # (used by the install rules at the bottom of this file).
        configure_file(${ONNXRUNTIME_DYLIB} ${CMAKE_BINARY_DIR}/plugins/libonnxruntime.dylib COPYONLY)
    endif()
else()
    set(ONNXRUNTIME_SO ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.${ONNXRUNTIME_VERSION})
    if(EXISTS ${ONNXRUNTIME_SO})
        add_custom_command(TARGET MambaSuiteClap POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_SO}
                $<TARGET_FILE_DIR:MambaSuiteClap>/libonnxruntime.so.${ONNXRUNTIME_VERSION}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_SO}
                $<TARGET_FILE_DIR:MambaSuiteClap>/libonnxruntime.so
            COMMENT "Deploying libonnxruntime.so to $<TARGET_FILE_DIR:MambaSuiteClap>"
            VERBATIM
        )
        add_custom_command(TARGET MambaSuiteARA POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_SO}
                $<TARGET_FILE_DIR:MambaSuiteARA>/libonnxruntime.so.${ONNXRUNTIME_VERSION}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_SO}
                $<TARGET_FILE_DIR:MambaSuiteARA>/libonnxruntime.so
            COMMENT "Deploying libonnxruntime.so to $<TARGET_FILE_DIR:MambaSuiteARA>"
            VERBATIM
        )
        # Unversioned symlink-named copy in top-level build/plugins/
        # (used by the install rules at the bottom of this file).
        configure_file(${ONNXRUNTIME_SO} ${CMAKE_BINARY_DIR}/plugins/libonnxruntime.so COPYONLY)
    endif()
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
    add_custom_target(clara_test
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS MambaSuiteClap MambaSuiteARA
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
#         libonnxruntime.so
#         models/
#           mamba_rt.onnx
#           mamba_rt.onnx.data
#
# ARA standalone:
#   ${CMAKE_INSTALL_PREFIX}/lib/ara/MambaSuiteARA.so
#
# Bare CLAP binary (for hosts that scan bare .clap files):
#   ${CMAKE_INSTALL_PREFIX}/lib/clap/MambaSuiteClap.clap
#
# An installed plugin directory includes everything needed at runtime
# (plugin + runtime lib + model). The RUNPATH is $ORIGIN so all
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

# --- Runtime dependencies for CLAP bundle ---
if(WIN32)
    install(FILES ${ONNXRUNTIME_DLL}
        DESTINATION ${CLAP_BUNDLE_DIR}
        COMPONENT clap
    )
elseif(APPLE)
    install(FILES
            ${ONNXRUNTIME_DYLIB}
            ${CMAKE_BINARY_DIR}/plugins/libonnxruntime.dylib
        DESTINATION ${CLAP_BUNDLE_DIR}
        COMPONENT clap
    )
else()
    install(FILES
            ${ONNXRUNTIME_SO}
            ${CMAKE_BINARY_DIR}/plugins/libonnxruntime.so
        DESTINATION ${CLAP_BUNDLE_DIR}
        COMPONENT clap
    )
endif()

# --- Models for CLAP bundle ---
install(DIRECTORY ${CMAKE_SOURCE_DIR}/models/
    DESTINATION ${CLAP_BUNDLE_DIR}/models
    COMPONENT clap
    FILES_MATCHING PATTERN "*.onnx*"
)

# --- Runtime dependencies for ARA ---
if(WIN32)
    install(FILES ${ONNXRUNTIME_DLL}
        DESTINATION ${ARA_INSTALL_PREFIX}
        COMPONENT ara
    )
elseif(APPLE)
    install(FILES
            ${ONNXRUNTIME_DYLIB}
            ${CMAKE_BINARY_DIR}/plugins/libonnxruntime.dylib
        DESTINATION ${ARA_INSTALL_PREFIX}
        COMPONENT ara
    )
else()
    install(FILES
            ${ONNXRUNTIME_SO}
            ${CMAKE_BINARY_DIR}/plugins/libonnxruntime.so
        DESTINATION ${ARA_INSTALL_PREFIX}
        COMPONENT ara
    )
endif()

# --- Models for ARA ---
install(DIRECTORY ${CMAKE_SOURCE_DIR}/models/
    DESTINATION ${ARA_INSTALL_PREFIX}/models
    COMPONENT ara
    FILES_MATCHING PATTERN "*.onnx*"
)

# Convenience: `cmake --install . --component clap` for CLAP only
#              `cmake --install . --component ara` for ARA only
#              `cmake --install .` for both
