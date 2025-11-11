package com.atleastitworks.example_ndk_ml

// ============================================================================
// Import statements: UI, Audio, Permissions, and Threading components
// ============================================================================
import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import kotlin.concurrent.thread

// ============================================================================
// BUILD INSTRUCTIONS FOR TENSORFLOW LITE C LIBRARY
// ============================================================================
// This section documents how to build the native TensorFlow Lite library
// required for ML inference on Android. These are reference steps only.
//
// docker pull tensorflow/build:2.20-python3.11
// git clone https://github.com/tensorflow/tensorflow.git
// - Download NDK https://developer.android.com/ndk/downloads extract to "android-ndk-r27d"
// - Whe you downloaded the Android studio, the SDK should be in your home dir
// docker run -v ~/Android:/home/Android -v ./android-ndk-r27d:/home/android-ndk-r27d-linux -v ./tensorflow:/home/tensorflow -it  tensorflow/build:2.20-python3.11 /bin/sh
//    cd /home/tensorflow
//    ./configure # select "y" when asked to compile for Androdid and pass NDK path: `/home/android-ndk-r27d-linux` and Android SDK path `/home/Android/Sdk`
//    bazel build -c opt --config=android_arm //tensorflow/lite/c:tensorflowlite_c
//    cp bazel-bin/tensorflow/lite/libtensorflowlite.so .
//    - (outside docker) Get the libtensorflowlite.so and copy to ./app/src/jniLibs/arm64-v8a/libtensorflowlite_c.so
//    bazel build -c opt --config=android_arm64 //tensorflow/lite/c:tensorflowlite_c
//    cp bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so .
//    - (outside docker) Get the libtensorflowlite_c.so and copy to ./app/src/jniLibs/armeabi-v7a/libtensorflowlite_c.so

// ============================================================================
// APPLICATION CONSTANTS
// ============================================================================
// These constants define the ML model's input requirements and classification
// thresholds for determining when to consider a prediction as valid.
object Constants {
    // MODEL_INPUT_LEN: The audio buffer size expected by the TensorFlow Lite model.
    // The model expects exactly 512 audio samples per inference. This is a fixed
    // requirement of the trained ML model - it won't work with different sizes.
    const val MODEL_INPUT_LEN = 512
    
    // MIN_RMS_VAL: Minimum Root Mean Square (RMS) threshold for audio detection.
    // RMS is a measure of audio signal loudness/energy. Below 0.005, the audio
    // is considered silence and won't be processed by the ML model to save CPU.
    const val MIN_RMS_VAL = 0.005
    
    // MIN_CLASSIFICATION_VAL: Minimum confidence score for accepting a prediction.
    // The ML model outputs confidence scores (0.0 to 1.0) for each class.
    // Only predictions with confidence > 0.75 are shown to the user.
    // This prevents false positives and noisy predictions.
    const val MIN_CLASSIFICATION_VAL = 0.75
}

// ============================================================================
// MAIN ACTIVITY: Audio ML Classification Application
// ============================================================================
// This Activity is the main entry point of the application. It handles:
// - UI interactions (start/stop buttons, result display)
// - Android permissions for microphone access
// - Audio capture from the device microphone
// - Delegation of ML inference to the native C++ processor
// - Real-time display of classification results
class MainActivity : AppCompatActivity() {

    // ========================================================================
    // INSTANCE VARIABLES
    // ========================================================================
    
    // Wrapper around the native C++ ML processor. Handles all audio inference
    // through JNI (Java Native Interface) calls to the TensorFlow Lite library.
    private lateinit var mlProcessor: NativeMLProcessor
    
    // UI Components: Button to start audio recording and classification
    private lateinit var startButton: Button
    
    // UI Components: Button to stop audio recording and classification
    private lateinit var stopButton: Button
    
    // UI Components: Text view to display classification results and messages to the user
    private lateinit var resultText: TextView

    // UI Components: Image view to display classification
    private lateinit var numImage: ImageView
    
    // Flag that tracks whether audio recording is currently active.
    // Used to control the recording loop in the background thread.
    private var isRecording = false

    // ========================================================================
    // COMPANION OBJECT: Static initialization
    // ========================================================================
    companion object {
        // Request code for Android's permission dialog. Arbitrary number used
        // to identify which permission request completed in onRequestPermissionsResult().
        private const val REQUEST_RECORD_AUDIO_PERMISSION = 200
        
        // Static initializer: Loads the native C++ library at app startup.
        // This makes all JNI functions available via System.loadLibrary().
        // The library is compiled from app/src/main/cpp/example_ndk_ml.cpp
        init {
            System.loadLibrary("example_ndk_ml")
        }
    }

    // ========================================================================
    // ASSET MANAGEMENT
    // ========================================================================
    /**
     * Copies a binary asset file from the app bundle to the internal storage.
     * 
     * Why this is needed: Android app assets are packed in the APK and can only
     * be read sequentially. TensorFlow Lite requires random access to the model
     * file, so we must copy it to internal storage (app cache) first.
     * 
     * @param assetName The name of the asset file in app/src/main/assets/
     * @param destPath The destination path in internal app storage
     */
    private fun copyAssetToFile(assetName: String, destPath: String) {
        // Open asset as input stream and destination file as output stream.
        // The .use {} block ensures streams are closed automatically.
        assets.open(assetName).use { input ->
            java.io.File(destPath).outputStream().use { output ->
                // Copy all bytes from asset to the destination file
                input.copyTo(output)
            }
        }
    }

    // ========================================================================
    // ACTIVITY LIFECYCLE: onCreate
    // ========================================================================
    /**
     * Called when the activity is first created. Initializes UI components,
     * loads the ML model, and sets up event listeners.
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Enable edge-to-edge display (extends content under system bars for
        // a more modern look). Requires handling of system insets manually.
        enableEdgeToEdge()
        
        // Load the UI layout from XML resource file (activity_main.xml)
        setContentView(R.layout.activity_main)
        
        // Handle system insets (status bar, navigation bar) to prevent content
        // from being hidden behind them. Adds padding to the main view.
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        // ====================================================================
        // UI COMPONENT INITIALIZATION
        // ====================================================================
        // Find and cache references to UI elements for later interaction
        startButton = findViewById(R.id.startButton)
        stopButton = findViewById(R.id.stopButton)
        resultText = findViewById(R.id.resultText)
        numImage = findViewById(R.id.numberImage)

        // ====================================================================
        // ML MODEL INITIALIZATION
        // ====================================================================
        resultText.text = "Loading Model..."

        // The model file "conv-classifier-model.tflite" is stored in the APK's
        // assets folder. We copy it to internal storage so TensorFlow Lite can
        // access it properly (assets are sequential-access only).
        val modelPath = "${filesDir.absolutePath}/conv-classifier-model.tflite"
        copyAssetToFile("conv-classifier-model.tflite", modelPath)
        
        // Create the native ML processor and pass the model path.
        // This initializes the TensorFlow Lite interpreter in the native C++ code.
        // If initialization fails, NativeMLProcessor throws an exception.
        mlProcessor = NativeMLProcessor(modelPath)

        resultText.text = "Model loaded, classifier initialized!"
        // ====================================================================
        // UI EVENT LISTENERS
        // ====================================================================
        // Start button: Check microphone permission, then begin recording
        startButton.setOnClickListener {
            if (checkPermissions()) {
                startRecording()
            }
        }

        // Stop button: Stop the recording loop and clean up resources
        stopButton.setOnClickListener {
            stopRecording()
        }

        // Initially disable the stop button (only enable it during recording)
        stopButton.isEnabled = false
    }

    // ========================================================================
    // PERMISSION HANDLING
    // ========================================================================
    /**
     * Checks if the app has permission to record audio from the microphone.
     * 
     * Android 6.0+ (API 23+) requires runtime permissions to be requested
     * at runtime, not just at install time. This function checks if we have
     * the RECORD_AUDIO permission and requests it if we don't.
     * 
     * @return true if permission is already granted, false if we just requested it
     */
    private fun checkPermissions(): Boolean {
        // Check if RECORD_AUDIO permission is already granted
        return if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            // Permission not granted: show system permission dialog
            // The response will be handled in onRequestPermissionsResult()
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                REQUEST_RECORD_AUDIO_PERMISSION
            )
            // Return false: caller should wait for onRequestPermissionsResult()
            false
        } else {
            // Permission already granted: return true to proceed
            true
        }
    }


    // ========================================================================
    // AUDIO RECORDING AND ML INFERENCE
    // ========================================================================
    /**
     * Starts recording audio from the microphone and performs real-time ML
     * classification in a background thread.
     * 
     * This function:
     * 1. Updates UI state (disable start button, enable stop button)
     * 2. Launches a background thread for audio processing
     * 3. Continuously reads audio from the microphone
     * 4. Sends audio buffers to the native ML processor
     * 5. Updates the UI with classification results
     * 6. Runs until the stop button is pressed
     */
    private fun startRecording() {
        // Set the recording flag to true; used to control the recording loop
        isRecording = true
        
        // Update UI: disable start button to prevent multiple recordings
        startButton.isEnabled = false
        
        // Update UI: enable stop button so user can stop recording
        stopButton.isEnabled = true
        
        // Update UI: show recording status to user
        resultText.text = "Recording..."

        // Launch background thread to handle audio recording (non-blocking).
        // Running audio processing on the main thread would freeze the UI.
        thread @androidx.annotation.RequiresPermission(android.Manifest.permission.RECORD_AUDIO) {
            // ================================================================
            // AUDIO CONFIGURATION
            // ================================================================
            
            // Sample rate: 44100 Hz is a standard audio rate for Android.
            // Matches common MP3/AAC quality and the model's expectations.
            val sampleRate = 44100
            
            // Buffer size calculation: AudioRecord needs a minimum buffer to
            // efficiently capture audio. We request the minimum and multiply by 2
            // to add a safety margin for smooth recording.
            var bufferSize = AudioRecord.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            ) * 2

            // Ensure buffer is at least the size the ML model expects.
            // The model requires exactly 512 samples per inference.
            // If bufferSize is smaller, we can't provide valid input to the model.
            if (bufferSize < Constants.MODEL_INPUT_LEN) {
                bufferSize = Constants.MODEL_INPUT_LEN
            }

            // ================================================================
            // PERMISSION DOUBLE-CHECK
            // ================================================================
            // Even though we checked permissions in onCreate(), they could have
            // been revoked before this thread started. Check again to be safe.
            if (ContextCompat.checkSelfPermission(
                    this@MainActivity,
                    Manifest.permission.RECORD_AUDIO
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                // Permission not available: display error and stop recording
                runOnUiThread {
                    resultText.text = "Permission denied"
                }
                isRecording = false
                startButton.isEnabled = true
                stopButton.isEnabled = false
                return@thread
            }

            // ================================================================
            // AUDIO RECORDING SETUP
            // ================================================================
            // Create AudioRecord object to capture audio from the microphone.
            // Parameters:
            // - MIC: capture from device microphone
            // - sampleRate: samples per second (44100)
            // - CHANNEL_IN_MONO: capture single channel (not stereo)
            // - ENCODING_PCM_16BIT: 16-bit signed PCM (standard audio format)
            // - bufferSize: internal Android buffer size
            val audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )

            // Begin capturing audio from the microphone
            audioRecord.startRecording()
            
            // Buffer to hold audio samples from one read() call.
            // Stores 16-bit PCM samples (short = 16 bits in Java).
            // Size is bufferSize/2 because each short is 2 bytes.
            val audioBuffer = ShortArray(bufferSize / 2)

            // Log: mark the start of the classification loop
            Log.i("MAIN", "Entering classification loop")
            
            // ================================================================
            // REAL-TIME CLASSIFICATION LOOP
            // ================================================================
            // Continuously read audio and perform inference until stopped
            while (isRecording) {
                // Read up to audioBuffer.size samples from the microphone.
                // Returns the number of samples actually read, or -1 on error.
                val readSize = audioRecord.read(audioBuffer, 0, audioBuffer.size)

                // Only process if we successfully read audio samples
                if (readSize > 0) {
                    // ============================================================
                    // STEP 1: Calculate audio loudness (RMS - Root Mean Square)
                    // ============================================================
                    val rmsValue = calculateRMS(audioBuffer)
                    // Optional debug: Log the RMS value
                    // Log.i("MAIN", "RMS Value: $rmsValue")

                    // ============================================================
                    // STEP 2: Check if audio is loud enough to process
                    // ============================================================
                    // If RMS is below threshold, it's likely silence.
                    // Skip ML inference to save CPU power.
                    if (rmsValue > Constants.MIN_RMS_VAL) {
                        // ======================================================
                        // STEP 3: Send audio to ML processor
                        // ======================================================
                        // Call the native C++ function to perform inference.
                        // processAudio() returns an array of confidence scores
                        // (one score per possible classification class).
                        val result = mlProcessor.processAudio(audioBuffer)
                        
                        // Find the class with the highest confidence score
                        val maxIndex = result.indices.maxByOrNull { result[it] }
                        
                        // If we found a valid result
                        if (maxIndex != null) {
                            // Log the prediction for debugging
                            // Log.i("MAIN", "Classification Value: ${result[maxIndex]} ($maxIndex). RMS Value: $rmsValue")

                            // ================================================
                            // STEP 4: Check confidence threshold
                            // ================================================
                            // Only show predictions with high confidence to reduce
                            // false positives and noisy classifications.
                            if (result[maxIndex] > Constants.MIN_CLASSIFICATION_VAL) {
                                // High confidence: show the prediction to user
                                runOnUiThread {
                                    // Show prediction data
                                    resultText.text = "Classification Value: ${result[maxIndex]} (idx: $maxIndex). RMS Value: $rmsValue"
                                    // Assign the corresponding image
                                    when (maxIndex) {
                                        0 -> numImage.setImageResource(R.drawable.icon_0)
                                        1 -> numImage.setImageResource(R.drawable.icon_1)
                                        2 -> numImage.setImageResource(R.drawable.icon_2)
                                        3 -> numImage.setImageResource(R.drawable.icon_3)
                                        4 -> numImage.setImageResource(R.drawable.icon_4)
                                        5 -> numImage.setImageResource(R.drawable.icon_5)
                                        6 -> numImage.setImageResource(R.drawable.icon_6)
                                        7 -> numImage.setImageResource(R.drawable.icon_7)
                                        8 -> numImage.setImageResource(R.drawable.icon_8)
                                        9 -> numImage.setImageResource(R.drawable.icon_9)
                                        10 -> numImage.setImageResource(R.drawable.icon_10)
                                        11 -> numImage.setImageResource(R.drawable.icon_11)
                                    }

                                }
                            } else {
                                // Low confidence: don't confuse user with noise
                                runOnUiThread {
                                    resultText.text = "No confidence."
                                    numImage.setImageResource(R.drawable.idle)
                                }
                            }
                        }

                    } else {
                        // Audio is too quiet: show "No signal" message
                        runOnUiThread {
                            resultText.text = "No signal."
                            numImage.setImageResource(R.drawable.idle)
                        }
                    }
                }
            }
            // Log: mark the end of the classification loop
            Log.i("MAIN", "Exiting classification loop")

            // ================================================================
            // AUDIO RECORDING CLEANUP
            // ================================================================
            // Stop capturing audio from the microphone
            audioRecord.stop()
            
            // Release all resources held by AudioRecord
            audioRecord.release()
        }
    }

    // ========================================================================
    // AUDIO ANALYSIS
    // ========================================================================
    /**
     * Calculates the Root Mean Square (RMS) of audio samples.
     * 
     * RMS is a measure of average audio signal loudness/energy.
     * Values range from 0.0 (silence) to 1.0 (maximum loudness).
     * 
     * Formula:
     *   RMS = sqrt(sum(samples[i]^2) / num_samples)
     * 
     * This is used to detect if there's meaningful audio (non-silence) before
     * sending the buffer to the ML model, which saves CPU power.
     * 
     * @param audioData Array of 16-bit PCM audio samples
     * @return RMS value between 0.0 and 1.0
     */
    private fun calculateRMS(audioData: ShortArray): Double {
        // Accumulator for sum of squared samples
        var sum = 0.0
        
        // Iterate through all audio samples
        for (sample in audioData) {
            // Normalize the 16-bit signed value to -1.0 to 1.0 range
            // Java shorts are 16-bit signed integers: range -32768 to 32767
            // Dividing by 32768.0 converts to float in [-1.0, 1.0]
            val normalized = sample / 32768.0
            
            // Add the square of the normalized sample to accumulate
            sum += normalized * normalized
        }
        
        // Return the square root of the mean (average) of squared values
        return kotlin.math.sqrt(sum / audioData.size)
    }

    // ========================================================================
    // RECORDING CONTROL
    // ========================================================================
    /**
     * Stops the audio recording and classification loop.
     * 
     * Sets the isRecording flag to false, which causes the background thread's
     * while loop to exit. The thread will then clean up audio resources and
     * terminate naturally.
     */
    private fun stopRecording() {
        // Signal the recording thread to exit its loop
        isRecording = false
        
        // Update UI: re-enable start button so user can start again
        startButton.isEnabled = true
        
        // Update UI: disable stop button (no recording to stop)
        stopButton.isEnabled = false
        
        // Update UI: show stopped status to user
        resultText.text = "Stopped"

        // Update UI: show idle image to user
        numImage.setImageResource(R.drawable.idle)
    }

    // ========================================================================
    // ACTIVITY LIFECYCLE: onDestroy
    // ========================================================================
    /**
     * Called when the activity is being destroyed (app closed or activity
     * is replaced). Performs cleanup of resources to prevent memory leaks.
     */
    override fun onDestroy() {
        // Call parent class cleanup first
        super.onDestroy()
        
        // Clean up the native ML processor: releases TensorFlow Lite resources,
        // deallocates memory, and closes the model file handle.
        mlProcessor.close()
    }

}