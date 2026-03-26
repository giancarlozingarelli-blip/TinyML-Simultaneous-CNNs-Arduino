/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "micro_features_micro_features_generator.h"

#include <cmath>
#include <cstring>

#include "micro_features_micro_model_settings.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
#include "tensorflow/lite/micro/micro_log.h"

// Configure FFT to output 16 bit fixed point.
#define FIXED_POINT 16

namespace {

FrontendState g_micro_features_state;
bool g_is_first_time = true;

}  // namespace

TfLiteStatus InitializeMicroFeatures() {
  FrontendConfig config;
  config.window.size_ms = kFeatureSliceDurationMs;
  config.window.step_size_ms = kFeatureSliceStrideMs;
  config.noise_reduction.smoothing_bits = 10;
  config.filterbank.num_channels = kFeatureSliceSize;
  config.filterbank.lower_band_limit = 125.0;
  config.filterbank.upper_band_limit = 7500.0;
  config.noise_reduction.smoothing_bits = 10;
  config.noise_reduction.even_smoothing = 0.025;
  config.noise_reduction.odd_smoothing = 0.06;
  config.noise_reduction.min_signal_remaining = 0.05;
  config.pcan_gain_control.enable_pcan = 1;
  config.pcan_gain_control.strength = 0.95;
  config.pcan_gain_control.offset = 80.0;
  config.pcan_gain_control.gain_bits = 21;
  config.log_scale.enable_log = 1;
  config.log_scale.scale_shift = 6;
  if (!FrontendPopulateState(&config, &g_micro_features_state,
                             kAudioSampleFrequency)) {
    MicroPrintf("FrontendPopulateState() failed");
    return kTfLiteError;
  }
  g_is_first_time = true;
  return kTfLiteOk;
}

// This is not exposed in any header, and is only used for testing, to ensure
// that the state is correctly set up before generating results.
void SetMicroFeaturesNoiseEstimates(const uint32_t* estimate_presets) {
  for (int i = 0; i < g_micro_features_state.filterbank.num_channels; ++i) {
    g_micro_features_state.noise_reduction.estimate[i] = estimate_presets[i];
  }
}

// Dichiara le variabili globali che verranno catturate nel file principale (.ino)
extern float g_input_scale;
extern int g_input_zero_point;

TfLiteStatus GenerateMicroFeatures(const int16_t* input, int input_size,
                                   int output_size, int8_t* output,
                                   size_t* num_samples_read) {
  const int16_t* frontend_input;
  if (g_is_first_time) {
    frontend_input = input;
    g_is_first_time = false;
  } else {
    frontend_input = input;
  }
  FrontendOutput frontend_output = FrontendProcessSamples(
      &g_micro_features_state, frontend_input, input_size, num_samples_read);

  for (size_t i = 0; i < frontend_output.size; ++i) {
    
    // 1. Riproduce la divisione standard del MicroFrontend Python (lo / 25.6)
    float float_val = frontend_output.values[i] / 25.6f;
    
    // 2. Quantizza il valore USANDO I PARAMETRI REALI del tuo modello .tflite
    int32_t value = std::round(float_val / g_input_scale) + g_input_zero_point;
    
    // 3. Applica il clipping di sicurezza per i numeri a 8-bit (-128 a 127)
    if (value < -128) {
      value = -128;
    }
    if (value > 127) {
      value = 127;
    }
    
    output[i] = static_cast<int8_t>(value);
  }

  return kTfLiteOk;
}