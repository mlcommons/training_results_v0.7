#ifndef CC_EVAL_H
#define CC_EVAL_H

#include <string>

namespace minigo {
namespace eval {

double run(bool   resign_enabled,
           double resign_threshold,
           uint64_t seed,
           int32_t virtual_losses,
           double value_init_penalty,
           std::string eval_model,
           std::string eval_device,
           int32_t num_eval_readouts,
           std::string target_model,
           std::string target_device,
           int32_t num_target_readouts,
           int32_t parallel_games,
           bool verbose);

} // namespace eval
} // namespace minigo

#endif // CC_EVAL_H
