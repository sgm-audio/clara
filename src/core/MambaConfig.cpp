// CLARA Mamba-3 Configuration — Runtime validation and serialization
//
// Part of the Mamba-1 → Mamba-3 migration (MAMBA3_MIGRATION_PLAN.md).
// ponytail: this file exists per migration plan manifest. Currently minimal —
// compile-time checks are in MambaConfig.h. Add weight-header validation
// here when the weight loader is implemented (Phase 3).

#include "core/MambaConfig.h"

namespace clara::mamba3 {

bool validate_config(const void* /*weight_header*/) {
    // Stub — will validate that a loaded weight file's header dimensions
    // match the compile-time constants in MambaConfig.h.
    return true;
}

}  // namespace clara::mamba3
