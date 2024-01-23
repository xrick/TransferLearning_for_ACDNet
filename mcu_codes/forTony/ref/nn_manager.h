#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

namespace{
    tflite::ErrorReporter* error_reporter = nullptr;
    tflite::MicroOpResolver* resolver = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    int inference_count;
    uint32_t inference_time;
    int32_t _input_number = 0;
    constexpr int kTensorArenaSize = 318169;//10 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
    NeuralNetworkFeatureProvider* featureProvider = nullptr;
    NeuralNetworkScores* scores = nullptr;

};

int soundsetup();
