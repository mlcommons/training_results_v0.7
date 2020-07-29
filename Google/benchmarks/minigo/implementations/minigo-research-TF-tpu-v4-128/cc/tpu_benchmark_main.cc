#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "base/commandlineflags.h"
#include "REDACTEDmemory/memory.h"
#include "REDACTEDstrings/str_cat.h"
#include "REDACTEDstrings/str_join.h"
#include "REDACTEDstrings/str_replace.h"
#include "REDACTEDstrings/str_split.h"
#include "REDACTEDsynchronization/mutex.h"
#include "REDACTEDtime/clock.h"
#include "REDACTEDtime/time.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/async/poll_thread.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/async/sharded_executor.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/async/thread.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/async/thread_safe_queue.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/file/path.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/file/utils.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/init.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/logging.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/model/model.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/model/types.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/platform/utils.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/tf_utils.h"
#include "third_party/tracing_framework_bindings_cpp/macros.h"

// Inference flags.
DEFINE_string(device, "",
              "ID of the device to run inference on. Can be left empty for "
              "single GPU machines. For a machine with N GPUs, a device ID "
              "should be specified in the range [0, N). For TPUs, pass the "
              "gRPC address for the device ID.");

DEFINE_string(model, "", "Path to a minigo model.");

DEFINE_int32(parallel_inference, 2, "Number of threads to run inference on.");
DEFINE_int32(batch_size, 8, "Batch size for inference.");
DEFINE_int64(num_inference, 10, "Run number of inferences for each worker.");

namespace {

using minigo::Model;
using minigo::ModelFactory;
using minigo::ModelInput;
using minigo::ModelOutput;
using minigo::Position;
using minigo::Thread;

class BenchmarkWorkerThread : public Thread {
 public:
  BenchmarkWorkerThread(ModelFactory* model_factory,
                        const std::string& model_path, int64 batch_size,
                        int64 num_inference)
      : batch_size_(batch_size), num_inference_(num_inference) {
    model_ = model_factory->NewModel(model_path);
  }

 private:
  void Run() override {
    // Setup model input and output.
    ModelInput input;
    ModelOutput output;
    input.sym = minigo::symmetry::kIdentity;
    Position position(minigo::Color::kBlack);
    input.position_history.push_back(&position);
    std::vector<const ModelInput*> inputs;
    std::vector<ModelOutput*> outputs;

    // The inputs aren't used by any of our test features so we don't need to
    // initialize them to anything meaningful.
    for (int i = 0; i < batch_size_; ++i) {
      inputs.push_back(&input);
      outputs.push_back(&output);
    }

    for (int64 i = 0; i < num_inference_; i++) {
      model_->RunMany(inputs, &outputs, &model_name_);
    }
  }
  std::string model_name_;
  std::unique_ptr<Model> model_;
  int64 batch_size_;
  int64 num_inference_;
};

}  // namespace

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  // Start the output threads.
  std::vector<std::unique_ptr<BenchmarkWorkerThread>> worker_threads;
  ModelDefinition def;
  def = LoadModelDefinition(FLAGS_model + ".minigo");
  auto* model_factory = GetModelFactory(def, FLAGS_device);
  worker_threads.reserve(FLAGS_parallel_inference);
  for (int i = 0; i < FLAGS_parallel_inference; ++i) {
    worker_threads.push_back(absl::make_unique<BenchmarkWorkerThread>(
        model_factory.get(), FLAGS_model, FLAGS_batch_size,
        FLAGS_num_inference));
  }
  for (auto& t : worker_threads) {
    t->Start();
  }

  for (auto& t : worker_threads) {
    t->Join();
  }

  return 0;
}
