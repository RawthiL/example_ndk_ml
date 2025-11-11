// ============================================================================
// TENSORFLOW LITE NATIVE AUDIO ML PROCESSOR - IMPLEMENTATION
// ============================================================================
// 
// This file implements the platform-independent ML processor for audio
// classification using TensorFlow Lite. It uses the TensorFlow Lite C API
// to perform inference without any Android-specific dependencies.
//
// Key characteristics:
// - Platform-agnostic: This same code compiles on Linux, macOS, Windows, etc.
// - Portable: Only depends on the TensorFlow Lite C library (no C++ API)
// - Efficient: Performs inference with minimal overhead
// - Thread-safe: Each MLProcessor instance is independent
// 
// =============================================================================

#include "ml_processor.h"
#include <cstdint>
#include <vector>
#include <cstdio>
#include <cmath>

// ============================================================================
// LOGGING MACROS (Platform-independent)
// ============================================================================

// Simple printf-based logging that works on all platforms
#define LOG_INFO(...) printf("[INFO] " __VA_ARGS__); printf("\n")
#define LOG_ERROR(...) printf("[ERROR] " __VA_ARGS__); printf("\n")

// ============================================================================
// ML PROCESSOR CLASS IMPLEMENTATION
// ============================================================================

/**
 * Constructor: Initialize the ML processor with a model file.
 * 
 * This constructor:
 * 1. Loads the .tflite model from the given file path
 * 2. Creates interpreter options (e.g., thread configuration)
 * 3. Creates a TensorFlow Lite interpreter from the model
 * 4. Allocates memory for input/output tensors
 * 5. Logs status messages
 * 
 * @param modelPath Absolute path to the .tflite model file on disk
 */
MLProcessor::MLProcessor(const char* modelPath) {
    // ====================================================================
    // STEP 1: Load Model File
    // ====================================================================
    // TfLiteModelCreateFromFile reads the .tflite file from disk and
    // parses its contents into a TfLiteModel structure that describes
    // the neural network architecture.
    model = TfLiteModelCreateFromFile(modelPath);
    if (!model) {
        LOG_ERROR("Failed to load model from %s", modelPath);
        return;
    }

    // ====================================================================
    // STEP 2: Create Interpreter Options
    // ====================================================================
    // Create a configuration object for the interpreter
    options = TfLiteInterpreterOptionsCreate();
    
    // Set number of CPU threads for inference
    // 2 threads provides good balance between speed and power consumption
    // More threads = faster inference but higher power usage
    TfLiteInterpreterOptionsSetNumThreads(options, 2);

    // ====================================================================
    // STEP 3: Create Interpreter
    // ====================================================================
    // Create a TfLiteInterpreter that will execute the model using the
    // given options. The interpreter is ready to accept input and produce
    // output after this step.
    interpreter = TfLiteInterpreterCreate(model, options);
    if (!interpreter) {
        LOG_ERROR("Failed to create interpreter");
        return;
    }

    // ====================================================================
    // STEP 4: Allocate Tensors
    // ====================================================================
    // Allocate memory for input and output tensors based on the model's
    // tensor requirements. This reserves GPU/CPU buffers for data.
    if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
        LOG_ERROR("Failed to allocate tensors");
    }

    // Log successful initialization
    LOG_INFO("Model loaded successfully from %s", modelPath);
}

/**
 * Destructor: Clean up all allocated resources.
 * 
 * Automatically called when the MLProcessor object is deleted.
 * Releases all memory and handles to prevent leaks.
 */
MLProcessor::~MLProcessor() {
    // Delete the interpreter (frees inference memory)
    if (interpreter) TfLiteInterpreterDelete(interpreter);
    
    // Delete the interpreter options (frees configuration memory)
    if (options) TfLiteInterpreterOptionsDelete(options);
    
    // Delete the model (frees model structure memory)
    if (model) TfLiteModelDelete(model);
}

/**
 * Process audio samples - fully portable implementation
 * 
 * This method:
 * 1. Validates the interpreter is initialized
 * 2. Gets the input tensor from the model
 * 3. Converts int16 PCM audio to normalized floats
 * 4. Copies data to the input tensor
 * 5. Invokes the interpreter (runs inference)
 * 6. Extracts and returns output predictions
 * 
 * @param audioData Raw audio samples (16-bit PCM)
 * @param length Number of samples in audioData
 * @return Vector of output predictions from the model
 */
std::vector<float> MLProcessor::processAudio(const int16_t* audioData, int length) {
    std::vector<float> result;

    if (!interpreter) {
        LOG_ERROR("Interpreter not initialized");
        return result;
    }

    // ====================================================================
    // STEP 1: Get Input Tensor
    // ====================================================================
    // Retrieve the input tensor from the interpreter. Index 0 refers to
    // the first (and typically only) input tensor in the model.
    TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    if (!inputTensor) {
        LOG_ERROR("Failed to get input tensor");
        return result;
    }

    // ====================================================================
    // STEP 2: Convert int16 to float and normalize
    // ====================================================================
    // Audio data comes as signed 16-bit integers. We need to:
    // 1. Convert to float (values roughly in range [-32768, 32767])
    // 2. Normalize to [-1.0, 1.0] by dividing by max amplitude
    
    std::vector<float> floatData(length);
    float maxAmplitude = 0.0f;
    for (int i = 0; i < MODEL_INPUT_LEN && i < length; i++) {
        floatData[i] = static_cast<float>(audioData[i]);
        // Track max amplitude to normalize between -1.0 and 1.0
        float absValue = std::abs(floatData[i]);
        if (absValue > maxAmplitude) {
            maxAmplitude = absValue;
        }
    }

    // Normalize to -1.0 to 1.0 range
    if (maxAmplitude > 0.0f) {
        for (int i = 0; i < MODEL_INPUT_LEN && i < length; i++) {
            floatData[i] = floatData[i] / maxAmplitude;
        }
    }

    LOG_INFO("Audio normalization - Max amplitude: %f", maxAmplitude);

    // ====================================================================
    // STEP 3: Copy Data to Input Tensor
    // ====================================================================
    // Copy the normalized float audio data into the input tensor buffer
    // that the interpreter will use for inference.
    TfLiteTensorCopyFromBuffer(inputTensor, floatData.data(),
                               MODEL_INPUT_LEN * sizeof(float));

    // ====================================================================
    // STEP 4: Run Inference
    // ====================================================================
    // Execute the neural network model with the input data.
    // This performs the forward pass through all layers of the network.
    if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
        LOG_ERROR("Failed to invoke interpreter");
        return result;
    }

    // ====================================================================
    // STEP 5: Extract Output Tensor
    // ====================================================================
    // Retrieve the output tensor from the interpreter. The model's final
    // layer produces output predictions that are stored here.
    const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(
            interpreter, 0);
    if (!outputTensor) {
        LOG_ERROR("Failed to get output tensor");
        return result;
    }

    // ====================================================================
    // STEP 6: Calculate Output Size
    // ====================================================================
    // Get the dimensions of the output tensor to determine how many
    // predictions were generated (e.g., 10 classes for digit recognition).
    int outputSize = 1;
    for (int i = 0; i < TfLiteTensorNumDims(outputTensor); i++) {
        outputSize *= TfLiteTensorDim(outputTensor, i);
    }

    // ====================================================================
    // STEP 7: Copy Output Data
    // ====================================================================
    // Copy the output predictions from the tensor into our result vector.
    result.resize(outputSize);
    TfLiteTensorCopyToBuffer(outputTensor, result.data(),
                             outputSize * sizeof(float));

    return result;
}
