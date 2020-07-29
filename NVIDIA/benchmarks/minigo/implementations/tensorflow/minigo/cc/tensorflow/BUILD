load("@org_tensorflow//tensorflow:tensorflow.bzl", "transitive_hdrs")

cc_library(
    name = "tensorflow",
    srcs = [
        "lib/libtensorflow_framework.so.1",
    ], 
    hdrs = [
        "include/tensorflow/core/common_runtime/gpu/gpu_init.h",
        "include/tensorflow/core/framework/graph.pb.h",
        "include/tensorflow/core/framework/tensor.h",
        "include/tensorflow/core/lib/core/errors.h",
        "include/tensorflow/core/lib/core/status.h",
        "include/tensorflow/core/lib/io/record_reader.h",
        "include/tensorflow/core/lib/io/record_writer.h",
        "include/tensorflow/core/platform/env.h",
        "include/tensorflow/core/platform/file_system.h",
        "include/tensorflow/core/platform/logging.h",
        "include/tensorflow/core/platform/protobuf.h",
        "include/tensorflow/core/public/session.h",
        "include/tensorflow/stream_executor/platform.h",
    ],

    includes = [
        "include",
        "include/tensorflow",
    ],
    linkopts = ["-lm"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "tf_lite",
    srcs = ["lib/libtensorflow_lite.so"],
    hdrs = glob(["include/tensorflow/lite/**"]),
    includes = ["include"],
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
    deps = ["@flatbuffers//:runtime_cc"],
)

# The following targets are used by cc/configure_tensorflow.sh to build the
# files required by the cc_library targets above.

# Compiling TensorFlow is handled by the :build rule's dependencies.
# The copy_outputs.sh script just copies the generated libraries and headers
# into the correct location under //cc/tensorflow/.
sh_binary(
    name = "build",
    srcs = ["copy_outputs.sh"],
    data = [
        ":libgrpc_runtime.so",
        ":libtensorflow_lite.so",
        ":tensorflow_hdrs",
        "@org_tensorflow//tensorflow:tensorflow_cc",
        "@org_tensorflow//tensorflow:tensorflow_framework",
        "@org_tensorflow//tensorflow/lite/toco",
        "@org_tensorflow//third_party/eigen3:eigen_third_party_header_files",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libgrpc_runtime.so",
    linkshared = True,
    linkstatic = True,
    deps = [
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_runtime",
    ],
)

cc_binary(
    name = "libtensorflow_lite.so",
    linkshared = True,
    linkstatic = True,
    deps = [
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

transitive_hdrs(
    name = "tensorflow_hdrs",
    deps = [
        ":libtensorflow_lite.so",
        "@com_google_protobuf//:protobuf_headers",
        "@eigen_archive//:eigen",
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/c:c_api_experimental",
        "@org_tensorflow//tensorflow/c/eager:c_api",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_session",
    ],
)
