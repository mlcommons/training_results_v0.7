// Custom init.cc that calls InitGoogle instead of OSS equivalents.

#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/init.h"

#include "base/init_google.h"
#include "REDACTEDdebugging/symbolize.h"

namespace minigo {

void Init(int* pargc, char*** pargv) {
  InitGoogle((*pargv)[0], pargc, pargv, true);
  absl::InitializeSymbolizer((*pargv)[0]);
}

}  // namespace minigo
