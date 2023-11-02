package com.example.itfd;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

@SuppressWarnings("ALL")
public class Classifier {
    private final Interpreter interpreter; //object of the interpreter class
    private final int[] inputShape; //shape of the input tensors
    private final int[] outputShape; //shape of the output tensors



    // Constructor for the Classifier class that loads the ML model and sets its input and output shapes
    public Classifier(AssetManager assetManager, String modelPath, int[] inputShape, int[] outputShape) {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);
        this.inputShape = inputShape;
        this.outputShape = outputShape;
        interpreter.resizeInput(0, inputShape); //resize the input tensor to match the input shape
    }

    // Method for running the loaded ML model on an input and returning the output
    public float[] classify(float[] input) {
        float[][] inputArr = new float[1][inputShape[0]];
        inputArr[0] = input;
        float[][] output = new float[2][outputShape[0]];
        interpreter.run(inputArr, output); // runs the loaded ML model on the input data
        return output[0];
    }
    // Method for loading the model file from assets and returning it to  loadModelFile
    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) {
        try {
            AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath); // Open an AssetFileDescriptor for the model file
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength); // Map the file channel to object and return it
        } catch (IOException e) {
            Log.e("Classifier", "Error loading model file: " + e.getMessage());
            return null;
        }
    }
}
