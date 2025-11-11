// ============================================================================
// TENSORFLOW LITE AUDIO ML PROCESSOR - HEADER
// ============================================================================
// 
// This header defines a platform-agnostic ML processor class that wraps
// TensorFlow Lite functionality for audio classification inference.
//
// Key characteristics:
// - Platform-agnostic: This same code compiles on Linux, macOS, Windows, etc.
// - Portable: Only depends on the TensorFlow Lite C library (no C++ API)
// - Efficient: Performs inference with minimal overhead
// - Thread-safe: Each MLProcessor instance is independent
// 
// =============================================================================

#ifndef ML_PROCESSOR_H
#define ML_PROCESSOR_H

#include <cstdint>
#include <vector>
#include "tensorflow/lite/c/c_api.h"

// ============================================================================
// MODEL CONSTANTS
// ============================================================================

// INPUT_LEN: Number of audio samples the model expects per inference.
// The trained TensorFlow Lite model requires exactly 512 samples as input.
// This is a hard constraint of the model architecture.
#define MODEL_INPUT_LEN 512

// ============================================================================
// ML PROCESSOR CLASS: TensorFlow Lite Wrapper
// ============================================================================
/**
 * Core audio processor class that wraps TensorFlow Lite functionality.
 * 
 * This class is platform-independent and can be used on any system that has
 * the TensorFlow Lite C library available (Android, Linux, Windows, macOS, etc.).
 * 
 * Responsibilities:
 * - Load ML model from file
 * - Create TensorFlow Lite interpreter with proper configuration
 * - Allocate tensors for inference
 * - Convert audio data to model input format
 * - Run inference (forward pass through the neural network)
 * - Extract and return predictions
 */
class MLProcessor {

// ========================================================================
// PRIVATE MEMBER VARIABLES
// ========================================================================
private:
    // TfLiteModel: Represents the loaded neural network model structure.
    // Loaded from the .tflite file and used to create interpreters.
    TfLiteModel* model;
    
    // TfLiteInterpreter: Executes inference by running the loaded model
    // on input tensors and producing output tensors.
    TfLiteInterpreter* interpreter;
    
    // TfLiteInterpreterOptions: Configuration settings for the interpreter.
    // Specifies number of threads, delegate options, etc.
    TfLiteInterpreterOptions* options;

// ========================================================================
// PUBLIC METHODS
// ========================================================================
public:
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
    MLProcessor(const char* modelPath);

    /**
     * Destructor: Clean up all allocated resources.
     * 
     * Automatically called when the MLProcessor object is deleted.
     * Releases all memory and handles to prevent leaks.
     */
    ~MLProcessor();

    /**
     * Process audio samples - fully portable implementation
     * @param audioData Raw audio samples (16-bit PCM)
     * @param length Number of samples
     * @return Vector of output predictions
     */
    std::vector<float> processAudio(const int16_t* audioData, int length);
};

#endif // ML_PROCESSOR_H
