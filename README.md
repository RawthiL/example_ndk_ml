# TensorFlow Lite Audio ML Classification - Android NDK Example

A complete Android application demonstrating real-time audio classification using TensorFlow Lite C API with native code (C++) via JNI (Java Native Interface).

## Overview

This project showcases:
- **Real-time audio processing** using Android's AudioRecord API
- **ML inference** with TensorFlow Lite C library
- **Native C++ implementation** compiled with CMake
- **JNI bridge** between Kotlin and native C++ code


## Project Structure

```
example_ndk_ml/
├── app/
│   └── src/
│       └── main/
│           ├── java/com/atleastitworks/example_ndk_ml/
│           │   ├── MainActivity.kt           # Main UI and audio recording
│           │   └── NativeMLProcessor.kt      # JNI wrapper
│           ├── cpp/
│           │   ├── CMakeLists.txt            # Build configuration
│           │   ├── ml_processor.h/.cpp       # TensorFlow Lite wrapper
│           │   └── jni_wrapper.cpp           # JNI bindings
│           ├── assets/
│           │   └── conv-classifier-model.tflite  # ML model
│           └── jniLibs/
│               ├── arm64-v8a/                # 64-bit ARM library
│               └── armeabi-v7a/              # 32-bit ARM library
├── build.gradle.kts
└── README.md
```

## Prerequisites

- Android Studio (latest version)
- Android NDK (version r27d or compatible)
- Android SDK (API level 24+)
- Docker (for building TensorFlow Lite C libraries): `arm64-v8a/conv-classifier-model.tflite` and `armeabi-v7a/conv-classifier-model.tflite`
- Git
- The model to load: [`conv-classifier-model.tflite`](https://github.com/RawthiL/pulsos_telefonicos)

## Building the TensorFlow Lite C Library

The native `.so` libraries need to be built, follow these steps:

### Step 1: Clone TensorFlow Repository

```bash
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
```

### Step 2: Prepare Build Environment

Download the [Android NDK](https://developer.android.com/ndk/downloads) and extract to a known location, e.g., `./android-ndk-r27d`

Locate your Android SDK (typically in your home directory: `~/Android/Sdk`):


### Step 3: Build with Docker (Recommended)

```bash
# Pull the TensorFlow build image
docker pull tensorflow/build:2.20-python3.11

# Run the Docker container with mounted volumes
# Replace paths as needed for your system
docker run -v ~/Android:/home/Android \
           -v ./android-ndk-r27d:/home/android-ndk-r27d-linux \
           -v ./tensorflow:/home/tensorflow \
           -it tensorflow/build:2.20-python3.11 /bin/sh

# Inside the container:
cd /home/tensorflow
./configure
# Select "y" for Android compilation
# Enter NDK path: /home/android-ndk-r27d-linux
# Enter SDK path: /home/Android/Sdk
```

### Step 4: Build for ARM Architectures

Inside the Docker container:

**For ARM64 (arm64-v8a):**
```bash
bazel build -c opt --config=android_arm64 //tensorflow/lite/c:tensorflowlite_c
cp bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so ../libtensorflowlite_c_arm64.so
```
The `.so` file will end up in `./tensorflow` (at the host).


**For ARM32 (armeabi-v7a):**
```bash
bazel build -c opt --config=android_arm //tensorflow/lite/c:tensorflowlite_c
cp bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so ../libtensorflowlite_c_arm32.so
```
The `.so` file will end up in `./tensorflow` (at the host).

### Step 5: Copy Libraries to Project

Outside the Docker container:
```bash
# Copy ARM64 library
cp libtensorflowlite_c_arm64.so app/src/main/jniLibs/arm64-v8a/libtensorflowlite_c.so

# Copy ARM32 library
cp libtensorflowlite_c_arm32.so app/src/main/jniLibs/armeabi-v7a/libtensorflowlite_c.so
```




## Copy TensorFlow Headers

Copy the TensorFlow headers to your project:

```bash
# Copy TensorFlow headers to include directory
cp -r tensorflow/tensorflow/lite/c/. app/src/main/cpp/include/tensorflow/lite/c/
cp -r tensorflow/tensorflow/lite/core/. app/src/main/cpp/include/tensorflow/lite/core/
cp -r tensorflow/tensorflow/lite/. app/src/main/cpp/include/tensorflow/lite/

# Also copy the common header
cp tensorflow/tensorflow/lite/common.h app/src/main/cpp/include/tensorflow/lite/
```

## Building the Project

1. **Open the project in Android Studio**

2. **Build the project**
   - Android Studio will automatically compile the C++ code using CMake
   - Gradle will build the Android app and link the native libraries

3. **Run on an emulator or device**
   - Select an Android device or emulator
   - Click "Run" (or press Shift+F10)

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────┐
│      Kotlin/Android (UI)            │
│  MainActivity + NativeMLProcessor   │
└──────────────┬──────────────────────┘
               │ JNI calls
               ▼
┌─────────────────────────────────────┐
│      C++ Native Code (jni_wrapper)  │
│  Handles JNI marshalling            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      ML Processor (ml_processor)    │
│  TensorFlow Lite C API              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   TensorFlow Lite C Library         │
│  (libtensorflowlite_c.so)           │
└─────────────────────────────────────┘
```

### Audio Processing Pipeline

1. **Audio Capture**: `MainActivity.kt` uses `AudioRecord` to capture audio from the microphone
2. **RMS Calculation**: Audio loudness is checked to avoid processing silence
3. **ML Inference**: Audio is passed to `NativeMLProcessor` via JNI
4. **Native Processing**: 
   - Audio samples are converted to the model's expected format
   - TensorFlow Lite interpreter runs the inference
   - Output predictions are extracted
5. **Result Filtering**: Only predictions with confidence > 0.75 are shown
6. **UI Update**: Classification results are displayed with corresponding icons

### Key Constants

These depend on the model you use!

- **MODEL_INPUT_LEN**: 512 audio samples per inference
- **MIN_RMS_VAL**: 0.005 (minimum loudness threshold)
- **MIN_CLASSIFICATION_VAL**: 0.75 (minimum confidence score)
- **SAMPLE_RATE**: 44100 Hz

## Application Usage

1. **Open the app** - The ML model will load (shows "Model loaded, classifier initialized!")
2. **Press "Start Recording"** - Audio recording begins and real-time classification starts
3. **Play a telephone dial tone** - The app classifies audio in real-time. Classification predictions and confidence scores are displayed with visual icons (key corresponding to dialed tone)
5. **Press "Stop Recording"** - Stops audio processing

## Important Notes

### Permissions

The app requires `RECORD_AUDIO` permission, which is requested at runtime (Android 6.0+). Grant the permission when prompted.

### TensorFlow Headers Structure

When copying TensorFlow headers to `cpp/include`, ensure the directory structure is preserved:

```
cpp/include/tensorflow/
├── lite/
│   ├── c/
│   │   ├── c_api.h
│   │   ├── c_api_opaque.h
│   │   └── ...
│   ├── core/
│   │   ├── common.h
│   │   └── ...
│   └── ...
└── ...
```

The `c_api.h` header is essential as it provides the TensorFlow Lite C API functions used in `ml_processor.cpp`.

### Memory Management

- The native `MLProcessor` is created once in `NativeMLProcessor.init{}`
- The same instance is reused for all audio inference
- Resources are cleaned up in `onDestroy()` via `mlProcessor.close()`

### Performance Optimization

- Audio processing runs in a background thread to prevent UI blocking
- Silent audio is skipped (RMS check) to save CPU cycles
- Model inference uses 2 threads for balanced performance/power consumption
- Confidence threshold filtering reduces false positives

## Troubleshooting

### Build Errors

**"Cannot find tensorflow/lite/c/c_api.h"**
- Ensure you've copied the TensorFlow headers to `cpp/include/`
- Verify the directory structure matches the expected layout

**"libtensorflowlite_c.so not found"**
- Ensure the `.so` files are in the correct `jniLibs` directories
- Verify file names match: `arm64-v8a/libtensorflowlite_c.so` and `armeabi-v7a/libtensorflowlite_c.so`

### Runtime Errors

**"Failed to initialize native ML processor"**
- Verify the model file path is correct
- Ensure the `.tflite` model exists in `assets/`
- Check that the model is compatible with TensorFlow Lite

**"Permission denied"**
- Grant microphone permission when prompted
- Check device settings: Settings → Apps → Permissions → Microphone

## References

- [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- [TensorFlow Lite C API](https://www.tensorflow.org/lite/guide/c_library)
- [Android NDK Documentation](https://developer.android.com/ndk)
- [Android AudioRecord API](https://developer.android.com/reference/android/media/AudioRecord)

## License

This project is provided as an example for educational purposes.

---

**Co-Authored**: `OpenCode Zen | Claude Haiku 4.5`
