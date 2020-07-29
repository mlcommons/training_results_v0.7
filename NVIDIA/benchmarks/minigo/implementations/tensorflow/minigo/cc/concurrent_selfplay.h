#ifndef CC_CONCURRENT_SELFPLAY_H
#define CC_CONCURRENT_SELFPLAY_H

#include <string>

namespace minigo {
namespace concurrent_selfplay {

void run (/* Inference flags */
          std::string n_model,
          int32_t cache_size_mb,
          int32_t cache_shards,

          /* Tree search flags */
          int32_t num_readouts,
          double fastplay_frequency,
          int32_t fastplay_readouts,
          int32_t virtual_losses,
          double dirichlet_alpha,
          double noise_mix,
          double value_init_penalty,
          bool target_pruning,
          double policy_softmax_temp,
          bool allow_pass,
          int32_t restrict_pass_alive_play_threshold,

          /* Threading flags. */
          int32_t num_selfplay_threads,
          int32_t num_parallel_search,
          int32_t num_parallel_inference,
          int32_t num_concurrent_games_per_thread,

          /* Game flags. */
          uint64_t seed,    
          double min_resign_threshold,
          double max_resign_threshold,
          double disable_resign_pct,
          int32_t num_games,
          bool run_forever,
          std::string abort_file,

          /* Output flags. */
          double holdout_pct,
          std::string output_dir,
          std::string holdout_dir,
          std::string sgf_dir,
          bool verbose,
          int32_t num_output_threads,

          /* Sample Fraction for output. */
          double sample_frac
	  ) ; 

} // namespace concurrent_selfplay
} // namespace minigo

#endif // CC_CONCURRENT_SELFPLAY_H
