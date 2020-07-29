/*!
 *  Copyright (c) 2017 by NVIDIA Corporation
 * \file nvtx.h
 * \brief Add support for NVTX
 */
#ifndef MXNET_COMMON_NVTX_H_
#define MXNET_COMMON_NVTX_H_

#if MXNET_USE_CUDA && MXNET_USE_NVTX
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvToolsExtCuda.h>
#include <vector>
#include <string>
#include <cstring>

class nvtx {
 public:
  // Palette of colors (make sure to add new colors to the vector in nameToColor()).
  static const uint32_t kRed     = 0xFF0000;
  static const uint32_t kGreen   = 0x00FF00;
  static const uint32_t kBlue    = 0x0000FF;
  static const uint32_t kYellow  = 0xB58900;
  static const uint32_t kOrange  = 0xCB4B16;
  static const uint32_t kRed1    = 0xDC322F;
  static const uint32_t kMagenta = 0xD33682;
  static const uint32_t kViolet  = 0x6C71C4;
  static const uint32_t kBlue1   = 0x268BD2;
  static const uint32_t kCyan    = 0x2AA198;
  static const uint32_t kGreen1  = 0x859900;

  static void gpuRangeStart(const uint32_t rgb, const char *range_name) {
    nvtxEventAttributes_t att;
    att.version = NVTX_VERSION;
    att.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    att.colorType = NVTX_COLOR_ARGB;
    att.color = rgb | 0xff000000;
    att.messageType = NVTX_MESSAGE_TYPE_ASCII;
    att.message.ascii = range_name;
    nvtxRangePushEx(&att);
  }

  // Utility to map a range name prefix to a random color based on its hash
  static uint32_t nameToColor(const char *range_name, int prefix_len) {
    static std::vector<uint32_t> colors{kRed, kGreen, kBlue, kYellow, kOrange, kRed1, kMagenta,
                                        kViolet, kBlue1, kCyan, kGreen1};
    std::string s(range_name, prefix_len);
    std::hash<std::string> hash_fn;
    return colors[hash_fn(s) % colors.size()];
  }

  // Utility to map a range name to a random color based on its hash
  static uint32_t nameToColor(const char *range_name) {
    return nameToColor(range_name, strlen(range_name));
  }

  static void gpuRangeStop() {
    nvtxRangePop();
  }
};

#endif  // MXNET_UDE_CUDA && MXNET_USE_NVTX
#endif  // MXNET_COMMON_NVTX_H_
