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

import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.util.Pair;
import android.util.Size;
import android.widget.Toast;

import com.example.mushroomclassifiersimple.databinding.ActivityMainBinding;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.firebase.ml.modeldownloader.CustomModel;
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions;
import com.google.firebase.ml.modeldownloader.DownloadType;
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MushroomClassifierSimpleTag";
    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static List<String> REQUIRED_PERMISSIONS = new ArrayList<>();
    private static final int WIDTH = 224;
    private static final int NUM_LABELS = 14;
    private static HashMap<Integer, String> NUM_TO_NAME_MAP = new HashMap<>();


    static {
        REQUIRED_PERMISSIONS.add(android.Manifest.permission.CAMERA);

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

        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS.toArray(new String[0]), REQUEST_CODE_PERMISSIONS);
        }

        cameraExecutor = Executors.newSingleThreadExecutor();

        getInterpreter();
        Log.d(TAG, "getInterpreter finished");
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

    private int predict(float[][][][] imageBuffer) {
        float[][] preds = new float[1][NUM_LABELS];
        interpreter.run(imageBuffer, preds);
        int maxPos = 0;

        for (int i = 0; i < preds[0].length; ++i) {
            if (preds[0][maxPos] < preds[0][i]) {
                maxPos = i;
            }
        }
//        Log.d(TAG, Arrays.toString(preds[0]));
        return maxPos;
    }

    private class MushroomClassifier implements ImageAnalysis.Analyzer {
        @Override
        public void analyze(@NonNull ImageProxy imageProxy) {
            if (interpreter == null) {
                imageProxy.close();
                return;
            }
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

                float[][][][] inputImage = new float[1][WIDTH][WIDTH][3];
                for (int i = 0; i < WIDTH; ++i) {
                    for (int j = 0; j < WIDTH; ++j) {
                        byte[] depthColumn = new byte[3];
                        cvMatRGB.get(i, j, depthColumn);

                        inputImage[0][i][j][0] = Byte.toUnsignedInt(depthColumn[0]);
                        inputImage[0][i][j][1] = Byte.toUnsignedInt(depthColumn[1]);
                        inputImage[0][i][j][2] = Byte.toUnsignedInt(depthColumn[2]);
                    }
                }

                int pred = predict(inputImage);

                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        viewBinding.predictionText.setText(NUM_TO_NAME_MAP.get(pred));
                    }
                });
//                Log.d(TAG, "Prediction: " + String.valueOf(pred));
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

    private void getInterpreter() {
        CustomModelDownloadConditions conditions = new CustomModelDownloadConditions.Builder()
                .requireWifi()  // Also possible: .requireCharging() and .requireDeviceIdle()
                .build();
        FirebaseModelDownloader.getInstance()
                .getModel("MushroomClassifier", DownloadType.LOCAL_MODEL_UPDATE_IN_BACKGROUND, conditions)
                .addOnSuccessListener(new OnSuccessListener<CustomModel>() {
                    @Override
                    public void onSuccess(CustomModel model) {
                        // Download complete. Depending on your app, you could enable the ML
                        // feature, or switch from the local model to the remote model, etc.

                        // The CustomModel object contains the local path of the model file,
                        // which you can use to instantiate a TensorFlow Lite interpreter.
                        Log.d(TAG, "onSuccess called");
                        File modelFile = model.getFile();
                        if (modelFile != null) {
                            Log.d(TAG, "modelFile available");
                            interpreter = new Interpreter(modelFile);
                            Log.d(TAG, "interpreter set");
                        }
                    }
                });
    }
}