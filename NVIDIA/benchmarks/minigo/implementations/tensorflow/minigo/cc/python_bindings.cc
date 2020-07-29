#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cc/sample_records.h"
#include "cc/eval.h"
#include "cc/concurrent_selfplay.h"

namespace py = pybind11;

PYBIND11_MODULE(minigo_python, m) {
  //SampleRecords
  m.def("run_sample_records", &::minigo::sample_records::run,
      py::arg("sample_frac") = 0.0,
      py::arg("num_records") = 0,
      py::arg("num_read_threads") = 1,
      py::arg("num_write_threads") = 1,
      py::arg("compression") = 1,
      py::arg("files_per_pattern") = 8192,
      py::arg("shuffle") = 0,
      py::arg("seed") = 0,
      py::arg("src_patterns") = "",
      py::arg("dst") = "",
      py::arg("verbose") = 0);

  //Eval
  m.def("run_eval", &::minigo::eval::run,
    py::arg("resign_enabled") = "True",
    py::arg("resign_threshold") = -0.999,
    py::arg("seed") = 0,
    py::arg("virtual_losses") = 8,
    py::arg("value_init_penalty") = 2.0,
    py::arg("eval_model") = "",
    py::arg("eval_device") = "",
    py::arg("num_eval_readouts") = 100,
    py::arg("target_model") = "",
    py::arg("target_device") = "",
    py::arg("num_target_readouts") = 100,
    py::arg("parallel_games") = 32,
    py::arg("verbose") = "False");

  //Selfplay
  m.def("run_concurrent_selfplay", &::minigo::concurrent_selfplay::run,
    py::arg("n_model") = "",
    py::arg("cache_size_mb") = 0,
    py::arg("cache_shards") = 8,

    py::arg("num_readouts") = 104,
    py::arg("fastplay_frequency") = 0.0,
    py::arg("fastplay_readouts") = 20,
    py::arg("virtual_losses") = 8,
    py::arg("dirichlet_alpha") = 0.03,
    py::arg("noise_mix") = 0.25,
    py::arg("value_init_penalty") = 2.0,
    py::arg("target_pruning") = "False",
    py::arg("policy_softmax_temp") = 0.98,
    py::arg("allow_pass") = "True",
    py::arg("restrict_pass_alive_play_threshold") = 4,

    py::arg("num_selfplay_threads") = 3,
    py::arg("num_parallel_search") = 3,
    py::arg("num_parallel_inference") = 3,
    py::arg("num_concurrent_games_per_thread") = 1,

    py::arg("seed") = 0,    
    py::arg("min_resign_threshold") = -1.0,
    py::arg("max_resign_threshold") = -0.8,
    py::arg("disable_resign_pct") = 0.1,
    py::arg("num_games") = 0,
    py::arg("run_forever") = "False",
    py::arg("abort_file") = "",

    py::arg("holdout_pct") = 0.03,
    py::arg("output_dir") = "",
    py::arg("holdout_dir") = "",
    py::arg("sgf_dir") = "" ,
    py::arg("verbose") = "False",
    py::arg("num_output_threads") = 1,

    py::arg("sample_frac") = 0.3 );

}

