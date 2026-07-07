# MinGW-w64 cross-compile toolchain for Windows x86_64
#
# Usage:
#   cmake -B build-windows \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/mingw64.cmake \
#     -DBUILD_TESTING=OFF
#   cmake --build build-windows --target MambaSuite_as_vst3 --parallel
#
# Output: build-windows/plugins/MambaSuite.dll (VST3 binary)
# Copy to Windows: %ProgramFiles%/Common Files/VST3/MambaSuite.vst3/

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
set(CMAKE_RC_COMPILER x86_64-w64-mingw32-windres)

set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32/sys-root/mingw)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Static link C++ runtime to avoid DLL dependencies on Windows
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static")

# Suppress GCC 16 warnings in clap-wrapper's bundled fmt library
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-dangling-reference -Wno-tautological-compare")
