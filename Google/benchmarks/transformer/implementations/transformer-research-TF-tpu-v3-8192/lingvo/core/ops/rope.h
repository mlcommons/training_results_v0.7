// absl::Cord is not yet open sourced. Introduce a Rope adapter class that's
// absl::Cord internally and std::string externally.

#ifndef LINGVO_CORE_OPS_ROPE_H_
#define LINGVO_CORE_OPS_ROPE_H_

#include "REDACTEDstrings/cord.h"

namespace tensorflow {
namespace babelfish {

typedef absl::Cord Rope;

}  // namespace babelfish
}  // namespace tensorflow

#endif  // LINGVO_CORE_OPS_ROPE_H_
