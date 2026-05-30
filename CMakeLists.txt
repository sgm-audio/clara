cmake_minimum_required(VERSION 3.14)
project(MambaAudioSuite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror=return-type")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -DNDEBUG -fno-rtti")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")

set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_RPATH "$ORIGIN")

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

# 4. Fetch ONNXRuntime (Precompiled binaries for Linux x64)
set(ONNXRUNTIME_VERSION 1.17.1)
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
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

# Common include directories
set(COMMON_INCLUDES
    ${clap_SOURCE_DIR}/include
    ${ara_sdk_SOURCE_DIR}
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Dispatch
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# Common sources (inference engine shared by both CLAP and ARA)
set(CORE_SOURCES
    src/core/MambaInferenceEngine.cpp
)

# --- CLAP Plugin ---
set(CLAP_SOURCES
    ${CORE_SOURCES}
    src/clap/MambaClapPlugin.cpp
)

add_library(MambaSuiteClap MODULE ${CLAP_SOURCES})
target_include_directories(MambaSuiteClap PRIVATE ${COMMON_INCLUDES})
set_target_properties(MambaSuiteClap PROPERTIES
    PREFIX ""
    SUFFIX ".clap"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
    INSTALL_RPATH "$ORIGIN"
    BUILD_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH ON
    CXX_VISIBILITY_PRESET hidden
)
target_link_libraries(MambaSuiteClap
    PRIVATE
    ${ONNXRUNTIME_LIB}
    pthread
    dl
)

# --- ARA Plugin ---
set(ARA_SOURCES
    ${CORE_SOURCES}
    src/ara/MambaAraController.cpp
    ${ara_sdk_SOURCE_DIR}/ARA_Library/PlugIn/ARAPlug.cpp
)

add_library(MambaSuiteARA MODULE ${ARA_SOURCES})
target_include_directories(MambaSuiteARA PRIVATE ${COMMON_INCLUDES})
set_target_properties(MambaSuiteARA PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
    INSTALL_RPATH "$ORIGIN"
    BUILD_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH ON
    CXX_VISIBILITY_PRESET hidden
)
target_link_libraries(MambaSuiteARA
    PRIVATE
    ${ONNXRUNTIME_LIB}
    pthread
    dl
)

# Copy models
if(EXISTS ${CMAKE_SOURCE_DIR}/models)
    file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR}/plugins)
endif()

# Deploy libonnxruntime.so alongside plugins so $ORIGIN resolves at runtime
set(ONNXRUNTIME_SO ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1.17.1)
if(EXISTS ${ONNXRUNTIME_SO})
    file(COPY ${ONNXRUNTIME_SO} DESTINATION ${CMAKE_BINARY_DIR}/plugins)
    configure_file(${ONNXRUNTIME_SO} ${CMAKE_BINARY_DIR}/plugins/libonnxruntime.so COPYONLY)
endif()
