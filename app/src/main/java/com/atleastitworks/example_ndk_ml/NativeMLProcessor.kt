package com.atleastitworks.example_ndk_ml

// ============================================================================
// NATIVE ML PROCESSOR: JNI Bridge to C++ TensorFlow Lite Implementation
// ============================================================================
/**
 * Kotlin wrapper class that provides a high-level interface to the native
 * C++ ML processor built with TensorFlow Lite.
 * 
 * This class uses JNI (Java Native Interface) to call C++ functions that
 * perform the actual ML inference. It abstracts away JNI complexity and
 * provides a clean Kotlin API for audio classification.
 * 
 * The native library is compiled from app/src/main/cpp/example_ndk_ml.cpp
 * and provides three main functions:
 * - nativeInit: Create and initialize a TensorFlow Lite interpreter
 * - nativeProcessAudio: Run inference on audio data
 * - nativeClose: Clean up resources
 */
class NativeMLProcessor(modelPath: String) {
    
    // ========================================================================
    // INSTANCE STATE
    // ========================================================================
    
    /**
     * Handle (pointer) to the native C++ MLProcessor object.
     * 
     * In C++, the MLProcessor is allocated on the heap and returned as a
     * pointer cast to a Long. We store this handle to use in subsequent
     * JNI calls. A value of 0 indicates the processor is not initialized.
     * 
     * Why store a handle instead of holding the object directly?
     * - Kotlin/Java can't directly hold C++ objects
     * - We pass the handle to each JNI function so C++ knows which
     *   processor instance to operate on
     * - Multiple independent processor instances can exist simultaneously
     */
    private var nativeHandle: Long = 0
    
    // ========================================================================
    // INITIALIZATION
    // ========================================================================
    
    /**
     * Constructor: Initialize the native ML processor with a model file.
     * 
     * This constructor calls nativeInit() via JNI to create a TensorFlow Lite
     * interpreter and load the ML model. Throws an exception if initialization
     * fails, so callers know immediately if something went wrong.
     * 
     * @param modelPath Absolute path to the TensorFlow Lite model file (.tflite)
     * @throws RuntimeException if the native processor fails to initialize
     */
    init {
        // Call JNI function to create and initialize the native processor.
        // Returns a handle (pointer cast to Long) or 0 on failure.
        nativeHandle = nativeInit(modelPath)
        
        // Verify initialization succeeded
        if (nativeHandle == 0L) {
            throw RuntimeException("Failed to initialize native ML processor")
        }
    }
    
    // ========================================================================
    // PUBLIC API
    // ========================================================================
    
    /**
     * Process audio samples using the TensorFlow Lite ML model.
     * 
     * Sends audio data to the native interpreter for inference.
     * The model processes the audio and returns confidence scores for each
     * possible classification class.
     * 
     * @param audioData Array of 16-bit PCM audio samples (typically 512 samples)
     * @return Array of floating-point confidence scores (one per class)
     *         Values range from 0.0 to 1.0
     * @throws IllegalStateException if the processor is not initialized
     */
    fun processAudio(audioData: ShortArray): FloatArray {
        // Safety check: ensure the processor was successfully initialized
        if (nativeHandle == 0L) {
            throw IllegalStateException("Native processor not initialized")
        }
        
        // Call JNI function to perform inference
        // Pass the handle so C++ knows which processor to use, and the audio data
        return nativeProcessAudio(nativeHandle, audioData)
    }
    
    /**
     * Clean up and release native resources.
     * 
     * Deallocates the TensorFlow Lite interpreter, releases memory, and
     * closes the model file. Should be called when the processor is no
     * longer needed to prevent memory leaks.
     * 
     * Safe to call multiple times; subsequent calls are no-ops.
     */
    fun close() {
        // Only clean up if the processor is initialized
        if (nativeHandle != 0L) {
            // Call JNI function to delete the native object and free memory
            nativeClose(nativeHandle)
            
            // Set handle to 0 to mark as closed
            // Prevents accidental use after close
            nativeHandle = 0
        }
    }
    
    /**
     * Finalizer: Ensures cleanup even if close() is not explicitly called.
     * 
     * Java's garbage collector calls finalize() before destroying an object.
     * This provides a safety net to release resources if the caller forgets
     * to call close(). However, relying on finalizers is not recommended for
     * production code - always call close() explicitly.
     */
    protected fun finalize() {
        close()
    }
    
    // ========================================================================
    // JNI FUNCTION DECLARATIONS
    // ========================================================================
    // These declarations tell Kotlin that these functions exist in the native
    // C++ library. The actual implementations are in example_ndk_ml.cpp.
    // The "external" keyword means the function is implemented outside Kotlin.
    
    /**
     * JNI Function: Initialize the native ML processor with a model file.
     * 
     * This function:
     * 1. Loads the TensorFlow Lite model from the given path
     * 2. Creates a TensorFlow Lite interpreter
     * 3. Allocates tensors for input and output
     * 4. Returns a pointer (as Long) to the MLProcessor object
     * 
     * @param modelPath Absolute path to the .tflite model file
     * @return Handle (pointer cast to Long) to the native MLProcessor, or 0 on failure
     */
    private external fun nativeInit(modelPath: String): Long

    /**
     * JNI Function: Process audio data through the ML model.
     * 
     * This function:
     * 1. Converts audio samples from 16-bit to float format
     * 2. Normalizes the audio to [-1.0, 1.0] range
     * 3. Copies data to the model's input tensor
     * 4. Runs the TensorFlow Lite inference
     * 5. Extracts the output tensor and converts to FloatArray
     * 
     * @param handle Handle to the native MLProcessor object (from nativeInit)
     * @param audioData Array of 16-bit PCM audio samples
     * @return Array of floating-point output values (model predictions)
     */
    private external fun nativeProcessAudio(handle: Long, audioData: ShortArray): FloatArray

    /**
     * JNI Function: Clean up and release native resources.
     * 
     * This function:
     * 1. Deletes the TensorFlow Lite interpreter
     * 2. Deletes the interpreter options
     * 3. Unloads the model file
     * 4. Frees all allocated memory
     * 
     * @param handle Handle to the native MLProcessor object (from nativeInit)
     */
    private external fun nativeClose(handle: Long): Unit

}
