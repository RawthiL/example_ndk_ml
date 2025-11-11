// ============================================================================
// TENSORFLOW LITE JNI WRAPPER FOR ANDROID
// ============================================================================
// 
// This file implements the JNI (Java Native Interface) bridge that allows
// Android Java/Kotlin code to call the platform-independent MLProcessor class.
//
// Key characteristics:
// - Android-specific: Contains all JNI binding code
// - Bridge layer: Converts between Java types and C++ types
// - Thread-safe: Each JNI call is independent
// - Minimal logic: Delegates actual work to MLProcessor
// 
// =============================================================================

// ============================================================================
// INCLUDES
// ============================================================================

// JNI header: Needed for calling Kotlin/Java functions and handling arguments
#include <jni.h>

// Standard C++ library headers
#include <string>        // String handling (e.g., const char*)
#include <vector>        // Dynamic arrays (for returning results)

// Android logging: Log output appears in Android Studio's Logcat
#include <android/log.h>

// Include the platform-independent ML processor
#include "ml_processor.h"

// ============================================================================
// LOGGING MACROS
// ============================================================================

// Tag for log messages: appears in Android logcat to identify source
#define LOG_TAG "AudioML"

// LOGI: Log informational messages (normal operation)
// Example: LOGI("Model loaded with %d inputs", numInputs);
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// LOGE: Log error messages (problems encountered)
// Example: LOGE("Failed to allocate tensors");
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ============================================================================
// JNI WRAPPER FUNCTIONS
// ============================================================================
// These functions bridge Java/Kotlin code to the C++ MLProcessor class.
// Each function converts Java types to C++ types, calls the appropriate
// MLProcessor method, and converts the result back to Java types.

extern "C" {

/**
 * JNI Function: Initialize ML processor with model file
 * 
 * Java signature:
 *   public native long nativeInit(String modelPath)
 * 
 * This function:
 * 1. Receives the model file path from Java
 * 2. Creates a new MLProcessor instance
 * 3. Returns a handle (pointer cast to long) to the Java caller
 * 
 * The handle is stored in Java and passed back to other JNI functions
 * to identify which MLProcessor instance to use.
 * 
 * @param env JNI environment pointer
 * @param this Reference to the calling object (unused here)
 * @param modelPath Java String containing path to .tflite model file
 * @return Long handle to MLProcessor instance (cast from pointer)
 */
JNIEXPORT jlong JNICALL
Java_com_atleastitworks_example_1ndk_1ml_NativeMLProcessor_nativeInit(
        JNIEnv* env, jobject /* this */, jstring modelPath) {

    // Convert Java string to C string
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (!path) {
        LOGE("Failed to get string from Java");
        return 0;
    }

    LOGI("Initializing MLProcessor with model: %s", path);

    // Create new MLProcessor instance with the model path
    MLProcessor* processor = new MLProcessor(path);
    
    // Release the C string (Java will manage the original)
    env->ReleaseStringUTFChars(modelPath, path);

    // Return pointer cast to jlong so Java can store it
    return reinterpret_cast<jlong>(processor);
}

/**
 * JNI Function: Process audio samples and get predictions
 * 
 * Java signature:
 *   public native float[] nativeProcessAudio(long handle, short[] audioData)
 * 
 * This function:
 * 1. Retrieves the MLProcessor instance from the handle
 * 2. Converts Java short array to C++ data
 * 3. Calls processAudio on the MLProcessor
 * 4. Converts C++ results back to Java float array
 * 
 * @param env JNI environment pointer
 * @param this Reference to the calling object (unused here)
 * @param handle MLProcessor pointer cast to jlong
 * @param audioData Java short array containing audio samples
 * @return Java float array containing model predictions
 */
JNIEXPORT jfloatArray JNICALL
Java_com_atleastitworks_example_1ndk_1ml_NativeMLProcessor_nativeProcessAudio(
        JNIEnv* env, jobject /* this */, jlong handle, jshortArray audioData) {

    // Cast handle back to MLProcessor pointer
    auto* processor = reinterpret_cast<MLProcessor*>(handle);
    if (!processor) {
        LOGE("Invalid processor handle");
        return env->NewFloatArray(0);
    }

    // Get the length of the input array
    jsize length = env->GetArrayLength(audioData);
    if (length <= 0) {
        LOGE("Empty audio data array");
        return env->NewFloatArray(0);
    }

    LOGI("Processing %d audio samples", length);

    // Get a C++ pointer to the Java short array data
    // JNI_ABORT means we don't copy changes back to Java (read-only)
    jshort* data = env->GetShortArrayElements(audioData, nullptr);
    if (!data) {
        LOGE("Failed to get array elements");
        return env->NewFloatArray(0);
    }

    // Call the platform-independent processAudio method
    std::vector<float> result = processor->processAudio(
            reinterpret_cast<const int16_t*>(data), length);

    // Release the array (no need to copy changes back)
    env->ReleaseShortArrayElements(audioData, data, JNI_ABORT);

    // Convert C++ vector result to Java float array
    jfloatArray output = env->NewFloatArray(result.size());
    if (output) {
        // Copy data from vector to Java array
        env->SetFloatArrayRegion(output, 0, result.size(), result.data());
    }

    LOGI("Returned %zu predictions", result.size());
    return output;
}

/**
 * JNI Function: Clean up and destroy ML processor
 * 
 * Java signature:
 *   public native void nativeClose(long handle)
 * 
 * This function:
 * 1. Retrieves the MLProcessor instance from the handle
 * 2. Deletes the instance (destructor cleans up TensorFlow Lite resources)
 * 3. Prevents memory leaks from native object
 * 
 * Must be called when done with the processor to release native memory.
 * 
 * @param env JNI environment pointer
 * @param this Reference to the calling object (unused here)
 * @param handle MLProcessor pointer cast to jlong
 */
JNIEXPORT void JNICALL
Java_com_atleastitworks_example_1ndk_1ml_NativeMLProcessor_nativeClose(
        JNIEnv* env, jobject /* this */, jlong handle) {

    // Cast handle back to MLProcessor pointer
    auto* processor = reinterpret_cast<MLProcessor*>(handle);
    if (processor) {
        LOGI("Closing MLProcessor");
        delete processor;  // Destructor will clean up all TensorFlow Lite resources
    } else {
        LOGE("Attempted to close invalid processor handle");
    }
}

} // extern "C"
