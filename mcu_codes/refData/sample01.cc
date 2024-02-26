TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
	// Define the input
	double x[16][16];	// Some input data which are numbers between -1. and 1. Import not shown for clarity

	// Set up logging
	tflite::MicroErrorReporter micro_error_reporter;

	// Model
	const tflite::Model* model = ::tflite::GetModel(g_mdl_model_data);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
	  TF_LITE_REPORT_ERROR(&micro_error_reporter,
	                         "Model provided is schema version %d not equal "
	                         "to supported version %d.\n",
	                         model->version(), TFLITE_SCHEMA_VERSION);
	}

	// Resolver
	tflite::AllOpsResolver resolver;

	// Arena
	constexpr int kTensorArenaSize = 500*1024;
	uint8_t tensor_arena[kTensorArenaSize];

	// Interpreter
	tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
	                                       kTensorArenaSize, &micro_error_reporter);

	interpreter.AllocateTensors();

	// Input
	TfLiteTensor* input = interpreter.input(0);

	// Does Input match
	TF_LITE_MICRO_EXPECT_NE(nullptr, input);
	TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
	TF_LITE_MICRO_EXPECT_EQ( 1, input->dims->data[0]);
	TF_LITE_MICRO_EXPECT_EQ(16, input->dims->data[1]);
	TF_LITE_MICRO_EXPECT_EQ(16, input->dims->data[2]);
	TF_LITE_MICRO_EXPECT_EQ( 1, input->dims->data[3]);
	TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, input->type);		//kTfLiteUInt8


	// Get quantization parameters
	double input_scale = input->params.scale;
	int input_zero_point = input->params.zero_point;


	// Quantize
	uint8_t x_quantized[256];
	for(int i=0;i<ni;i++){
		for(int j=0;j<nj;j++){
			x_quantized[i*nj+j] = (uint8_t) ((x[i][j]/input_scale) + input_zero_point);
			input->data.uint8[i*nj+j] = x_quantized[i*nj+j];
		}
	}


	// Run the model
	MicroPrintf("-----invoke-----");
	TfLiteStatus invoke_status = interpreter.Invoke();
	MicroPrintf("-----check-----");
	TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

	MicroPrintf("End of File");


}

TF_LITE_MICRO_TESTS_END