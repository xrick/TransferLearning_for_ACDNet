void setup(){
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  model = tflite::GetModel(g_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  {
    printf("MODEL:         %s\n", g_model_name);
    printf("MODEL SIZE:    %d\n", g_model_tflite_len);
    printf("FEATURE_WIDTH: %d\n", g_feature_size);
    printf("FEATURE_PATH:  %s\n", FEATURE_PATH);

    if (FEATURE_WIDTH != g_feature_size) {
        printf("Error: Expected FEATURE_WIDTH=%d, got %d", g_feature_size, FEATURE_WIDTH);
        exit(-1);
    }
  }

  {
    static tflite::MicroMutableOpResolver<8> static_resolver;
    
    static_resolver.AddConv2D();    
    static_resolver.AddDepthwiseConv2D();
    static_resolver.AddFullyConnected();
    static_resolver.AddReshape();
    static_resolver.AddSoftmax();
    static_resolver.AddAveragePool2D();
    static_resolver.AddMaxPool2D();
    static_resolver.AddTranspose();

    resolver = &static_resolver;
    printf("Network build done and Resolver done\n");
  }

  {
    static tflite::MicroInterpreter static_interpreter(
      model, *resolver, tensor_arena, kTensorArenaSize, error_reporter);

    interpreter = &static_interpreter;
    printf("Interpreter done\n");    

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }
    printf("Allocated tensors\n");
    printf("Arena used %u bytes\n", interpreter->arena_used_bytes());
    // printf("Interpreter State\n");
    // tflite::PrintInterpreterState(interpreter);
  }

  {
    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);

    printf("Dimensions\n");
    // We can view the input dimensions of the model
    printf("input->dims->size:    %u\n", input->dims->size);
    for (int i = 0; i < input->dims->size; i++) {
        printf("input->dims->data[%d]: %u\n", i, input->dims->data[i]);
    }

    // We can observe the expected data type of the input layer
    if (input->type == kTfLiteInt8) printf("int8_t input\n");
    else if (input->type == kTfLiteInt16) printf("int16_t input\n");
    else if (input->type == kTfLiteUInt8) printf("uint8_t input\n");
    else if (input->type == kTfLiteFloat16) printf("float16 input\n");
    else if (input->type == kTfLiteFloat32) printf("float32 input\n");    
    else printf("Unknown input type\n");

    printf("output->dims->size:    %u\n", output->dims->size);
    for (int i = 0; i < output->dims->size; i++) {
        printf("output->dims->data[%d]: %u\n", i, output->dims->data[i]);
    }

    // We can observe the expected data type of the input layer
    if (output->type == kTfLiteInt8) printf("int8_t output\n");
    else if (output->type == kTfLiteInt16) printf("int16_t output\n");
    else if (output->type == kTfLiteUInt8) printf("uint8_t output\n");
    else if (output->type == kTfLiteFloat16) printf("float16 output\n");
    else if (output->type == kTfLiteFloat32) printf("float32 output\n");    
    else printf("Unknown output type\n");

    // Keep track of how many inferences we have performed.
    inference_count = 0;
  }

}