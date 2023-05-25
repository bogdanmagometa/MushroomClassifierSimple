package com.example.mushroomclassifiersimple;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.media.Image;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.util.Log;
import android.util.Pair;
import android.util.Size;
import android.widget.Toast;

import com.example.mushroomclassifiersimple.databinding.ActivityMainBinding;
import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MushroomClassifierSimple";
    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static List<String> REQUIRED_PERMISSIONS = new ArrayList<>();
    private static final int WIDTH = 224;
    private static final int NUM_LABELS = 14;
    private static HashMap<Integer, String> NUM_TO_NAME_MAP = new HashMap<>();


    static {
        REQUIRED_PERMISSIONS.add(android.Manifest.permission.CAMERA);
        REQUIRED_PERMISSIONS.add(android.Manifest.permission.RECORD_AUDIO);
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            REQUIRED_PERMISSIONS.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }

        if (!OpenCVLoader.initDebug())
            Log.d("ERROR", "Unable to load OpenCV");
        else
            Log.d("SUCCESS", "OpenCV loaded");

        NUM_TO_NAME_MAP.put(0, "agaricus_xanthodermus");
        NUM_TO_NAME_MAP.put(1, "amanita_muscaria");
        NUM_TO_NAME_MAP.put(2, "amanita_phalloides");
        NUM_TO_NAME_MAP.put(3, "armillaria_mellea");
        NUM_TO_NAME_MAP.put(4, "boletus_edulis");
        NUM_TO_NAME_MAP.put(5, "cantharellus_cibarius");
        NUM_TO_NAME_MAP.put(6, "chalciporus_piperatus");
        NUM_TO_NAME_MAP.put(7, "hygrophoropsis_aurantiaca");
        NUM_TO_NAME_MAP.put(8, "hypholoma_fasciculare");
        NUM_TO_NAME_MAP.put(9, "inocybe_geophylla");
        NUM_TO_NAME_MAP.put(10, "rubroboletus_satanas");
        NUM_TO_NAME_MAP.put(11, "russula_emetica");
        NUM_TO_NAME_MAP.put(12, "russula_vesca");
        NUM_TO_NAME_MAP.put(13, "suillus_luteus");
    }

    ActivityMainBinding viewBinding;
    ExecutorService cameraExecutor;
    Interpreter interpreter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        viewBinding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(viewBinding.getRoot());

        if (Build.VERSION.SDK_INT >= 30) {
            if (!Environment.isExternalStorageManager()) {
                Intent getpermission = new Intent();
                getpermission.setAction(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION);
                startActivity(getpermission);
            }
        }

        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS.toArray(new String[0]), REQUEST_CODE_PERMISSIONS);
        }

        cameraExecutor = Executors.newSingleThreadExecutor();

        interpreter = new Interpreter(getModelByteBuffer());
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {
                ProcessCameraProvider cameraProvider;
                try {
                    cameraProvider = cameraProviderFuture.get();
                } catch (ExecutionException e) {
                    throw new RuntimeException(e);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(viewBinding.viewFinder.getSurfaceProvider());

                //TODO: WTF is RestrictedApi
                @SuppressLint("RestrictedApi") ImageAnalysis imageAnalysis = new ImageAnalysis.Builder().setImageQueueDepth(10)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setSupportedResolutions(Arrays.asList(Pair.create(ImageFormat.YUV_420_888, new Size[]{new Size(640, 480)}))) //TODO: check supported resolutions
                        .build();
                imageAnalysis.setAnalyzer(cameraExecutor, new MushroomClassifier());

                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

                try {
                    cameraProvider.unbindAll();
                    cameraProvider.bindToLifecycle(MainActivity.this, cameraSelector, preview, imageAnalysis);
                } catch (Exception e) {
                    Log.d(TAG, "Use case binding failed", e);
                }
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private Bitmap toBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer buffer = planes[0].getBuffer();
        int pixelStride = planes[0].getPixelStride();
        int rowStride = planes[0].getRowStride();
        int rowPadding = rowStride - pixelStride * image.getWidth();
        Bitmap bitmap = Bitmap.createBitmap(image.getWidth() + rowPadding / pixelStride,
                image.getHeight(), Bitmap.Config.ARGB_8888);
        bitmap.copyPixelsFromBuffer(buffer);
        return bitmap;
    }

    private int predict(ByteBuffer inputImage) {
        int bufferSize = NUM_LABELS * java.lang.Float.SIZE / java.lang.Byte.SIZE;
        ByteBuffer modelOutput = ByteBuffer.allocateDirect(bufferSize);//.order(ByteOrder.nativeOrder());
        interpreter.run(inputImage, modelOutput);
        int maxPos = 0;

        FloatBuffer probabilities = modelOutput.asFloatBuffer();

        for (int i = 0; i < probabilities.capacity(); ++i) {
            if (probabilities.get(maxPos) < probabilities.get(i)) {
                maxPos = i;
            }
        }
        return maxPos;
    }

    private class MushroomClassifier implements ImageAnalysis.Analyzer {
        @Override
        public void analyze(@NonNull ImageProxy imageProxy) {
            Image image = imageProxy.getImage();
            if (image != null) {
                Bitmap bitmap = toBitmap(image);
                Mat cvMatRGB = new Mat();
                Utils.bitmapToMat(bitmap, cvMatRGB);
                Imgproc.cvtColor(cvMatRGB, cvMatRGB, Imgproc.COLOR_RGBA2RGB);
                Core.rotate(cvMatRGB, cvMatRGB, Core.ROTATE_90_CLOCKWISE);
                Imgproc.resize(cvMatRGB, cvMatRGB, new org.opencv.core.Size(WIDTH, WIDTH));

                byte[] bytes = new byte[WIDTH * WIDTH * 3];
                cvMatRGB.get(0, 0, bytes);

//                float[][][][] inputImage = new float[1][WIDTH][WIDTH][3];
                ByteBuffer inputImage = ByteBuffer.allocateDirect(WIDTH * WIDTH * 3 * 4).order(ByteOrder.nativeOrder());
                for (int i = 0; i < WIDTH; ++i) {
                    for (int j = 0; j < WIDTH; ++j) {
                        byte[] depthColumn = new byte[3];
                        cvMatRGB.get(i, j, depthColumn);

                        inputImage.putFloat(Byte.toUnsignedInt(depthColumn[0]));
                        inputImage.putFloat(Byte.toUnsignedInt(depthColumn[1]));
                        inputImage.putFloat(Byte.toUnsignedInt(depthColumn[2]));
                    }
                }

                int pred = predict(inputImage);

                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        viewBinding.predictionText.setText(NUM_TO_NAME_MAP.get(pred));
                    }
                });
                Log.d(TAG, "Prediction: " + String.valueOf(pred));
            } //TODO: handle the other case
            imageProxy.close();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private ByteBuffer getModelByteBuffer() {
        // Obtain a reference to the Resources object
        Resources res = getResources();

        // Get the resource ID of the raw resource you want to read
        int resourceId = R.raw.model;

        // Open an InputStream to the raw resource
        InputStream inputStream = res.openRawResource(resourceId);

        // Create a new byte array to hold the data
        byte[] buffer = new byte[0];
        try {
            buffer = new byte[inputStream.available()];
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Read the data from the InputStream into the byte array
        try {
            inputStream.read(buffer);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Create a new ByteBuffer from the byte array
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(buffer.length);
        byteBuffer.put(buffer);

        return byteBuffer;
    }
}