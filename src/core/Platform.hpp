#pragma once

#include <fstream>
#include <string>

#define MAMBA_PROJECT_URL "https://github.com/scottmill306/CLARA"

#if defined(_WIN32)
#    define NOMINMAX
#    include <windows.h>
#    define MAMBA_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#    include <dlfcn.h>

#    include <cstdint>
#    if defined(__APPLE__)
#        include <mach-o/dyld.h>
#    endif
#    define MAMBA_EXPORT __attribute__((visibility("default")))
#endif

inline std::string mambaResolveModulePath(void *funcAddr, const char *modelFile) {
    (void) funcAddr;
    std::string base;

#if defined(_WIN32)
    char path[MAX_PATH];
    HMODULE hm = nullptr;
    if (funcAddr) {
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR) funcAddr, &hm);
    }
    if (hm && GetModuleFileNameA(hm, path, MAX_PATH)) {
        base = path;
    }
#else
    if (funcAddr) {
        Dl_info info;
        if (dladdr(funcAddr, &info) && info.dli_fname) {
            base = info.dli_fname;
        }
    }
#    if defined(__APPLE__)
    if (base.empty()) {
        char path[1024];
        uint32_t size = sizeof(path);
        if (_NSGetExecutablePath(path, &size) == 0) {
            base = path;
        }
    }
#    endif
#endif

    auto sep = base.rfind('/');
    if (sep == std::string::npos) sep = base.rfind('\\');
    if (sep != std::string::npos) base.resize(sep + 1);
    else
        base = "./";

    return base + "models/" + modelFile;
}

inline bool mambaFileExists(const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
}

inline std::string mambaResolveModelDir(void *funcAddr) {
    std::string base;

#if defined(_WIN32)
    char path[MAX_PATH];
    HMODULE hm = nullptr;
    if (funcAddr) {
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR) funcAddr, &hm);
    }
    if (hm && GetModuleFileNameA(hm, path, MAX_PATH)) {
        base = path;
    }
#else
    if (funcAddr) {
        Dl_info info;
        if (dladdr(funcAddr, &info) && info.dli_fname) {
            base = info.dli_fname;
        }
    }
#    if defined(__APPLE__)
    if (base.empty()) {
        char path[1024];
        uint32_t size = sizeof(path);
        if (_NSGetExecutablePath(path, &size) == 0) {
            base = path;
        }
    }
#    endif
#endif

    auto sep = base.rfind('/');
    if (sep == std::string::npos) sep = base.rfind('\\');
    if (sep != std::string::npos) base.resize(sep + 1);
    else
        base = "./";

    return base + "models/";
}
