// CLARA Mamba-3 Configuration — Runtime validation and serialization
//
// Compile-time checks are in MambaConfig.h. Runtime validation lives in
// WeightLoader::validate_header(). This file exists to satisfy the CMake
// source manifest and may grow when serialization/deserialization is needed.

#include "core/MambaConfig.h"

namespace clara::mamba3 {
// ponytail: stub removed — validation is in WeightLoader::validate_header()
} // namespace clara::mamba3
