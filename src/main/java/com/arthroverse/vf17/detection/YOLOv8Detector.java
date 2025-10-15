package com.arthroverse.vf17.detection;

import ai.onnxruntime.*;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class YOLOv8Detector {
    private OrtEnvironment env;
    private OrtSession session;
    private final int inputWidth = 640;
    private final int inputHeight = 640;
    private final float confThreshold = 0.25f;
    private final float iouThreshold = 0.45f;

    public YOLOv8Detector(String modelPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        session = env.createSession(modelPath, opts);
    }

    public List<Detection> detect(String imagePath) throws Exception {
        // Load and preprocess image
        BufferedImage image = ImageIO.read(new File(imagePath));
        float[] inputData = preprocessImage(image);

        // Create input tensor
        long[] shape = {1, 3, inputHeight, inputWidth};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env,
                FloatBuffer.wrap(inputData), shape);

        // Run inference
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(session.getInputNames().iterator().next(), inputTensor);

        OrtSession.Result results = session.run(inputs);

        // Process outputs
        float[][] output = processOutput(results);
        List<Detection> detections = postProcess(output, image.getWidth(),
                image.getHeight());

        // Cleanup
        inputTensor.close();
        results.close();

        return detections;
    }

    private float[] preprocessImage(BufferedImage image) {
        // Resize image to input dimensions
        BufferedImage resized = new BufferedImage(inputWidth, inputHeight,
                BufferedImage.TYPE_INT_RGB);
        resized.getGraphics().drawImage(image, 0, 0, inputWidth, inputHeight, null);

        // Convert to CHW format and normalize
        float[] data = new float[3 * inputHeight * inputWidth];
        int idx = 0;

        // RGB channels
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < inputHeight; h++) {
                for (int w = 0; w < inputWidth; w++) {
                    int rgb = resized.getRGB(w, h);
                    int channel = (c == 0) ? ((rgb >> 16) & 0xFF) :
                            (c == 1) ? ((rgb >> 8) & 0xFF) :
                                    (rgb & 0xFF);
                    data[idx++] = channel / 255.0f;  // Normalize to [0, 1]
                }
            }
        }
        return data;
    }

    private float[][] processOutput(OrtSession.Result results) throws OrtException {
        OnnxValue outputValue = results.get(0);
        float[][][] rawOutput = (float[][][]) outputValue.getValue();

        // YOLOv8 output shape is [1, 84, 8400] for COCO dataset
        // Transpose to [8400, 84]
        int numDetections = rawOutput[0][0].length;
        int numFeatures = rawOutput[0].length;

        float[][] transposed = new float[numDetections][numFeatures];
        for (int i = 0; i < numDetections; i++) {
            for (int j = 0; j < numFeatures; j++) {
                transposed[i][j] = rawOutput[0][j][i];
            }
        }
        return transposed;
    }

    private List<Detection> postProcess(float[][] output, int originalWidth,
                                        int originalHeight) {
        List<Detection> detections = new ArrayList<>();

        for (float[] detection : output) {
            // Extract box coordinates and confidence
            float x_center = detection[0];
            float y_center = detection[1];
            float width = detection[2];
            float height = detection[3];

            // Find class with highest confidence
            float maxConf = 0;
            int classId = 0;
            for (int i = 4; i < detection.length; i++) {
                if (detection[i] > maxConf) {
                    maxConf = detection[i];
                    classId = i - 4;
                }
            }

            if (maxConf > confThreshold) {
                // Convert to corner coordinates
                float x1 = (x_center - width / 2) * originalWidth / inputWidth;
                float y1 = (y_center - height / 2) * originalHeight / inputHeight;
                float x2 = (x_center + width / 2) * originalWidth / inputWidth;
                float y2 = (y_center + height / 2) * originalHeight / inputHeight;

                detections.add(new Detection(x1, y1, x2, y2, maxConf, classId));
            }
        }

        // Apply Non-Maximum Suppression
        return applyNMS(detections);
    }

    private List<Detection> applyNMS(List<Detection> detections) {
        detections.sort((a, b) -> Float.compare(b.confidence, a.confidence));
        List<Detection> result = new ArrayList<>();

        while (!detections.isEmpty()) {
            Detection best = detections.remove(0);
            result.add(best);

            detections.removeIf(det ->
                    det.classId == best.classId &&
                            calculateIoU(best, det) > iouThreshold);
        }

        return result;
    }

    private float calculateIoU(Detection a, Detection b) {
        float x1 = Math.max(a.x1, b.x1);
        float y1 = Math.max(a.y1, b.y1);
        float x2 = Math.min(a.x2, b.x2);
        float y2 = Math.min(a.y2, b.y2);

        float intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
        float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
        float union = areaA + areaB - intersection;

        return intersection / union;
    }

    public void close() throws OrtException {
        session.close();
        env.close();
    }

    // Detection class to hold results
    public static class Detection {
        public float x1, y1, x2, y2;
        public float confidence;
        public int classId;

        public Detection(float x1, float y1, float x2, float y2,
                         float confidence, int classId) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.confidence = confidence;
            this.classId = classId;
        }
    }

    public List<Detection> detect(Mat frame) throws OrtException {
        float[] inputData = preprocessMat(frame);

        // Create input tensor
        long[] shape = {1, 3, inputHeight, inputWidth};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env,
                FloatBuffer.wrap(inputData), shape);

        // Run inference
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(session.getInputNames().iterator().next(), inputTensor);

        OrtSession.Result results = session.run(inputs);

        // Process outputs
        float[][] output = processOutput(results);
        List<Detection> detections = postProcess(output, frame.width(), frame.height());

        // Cleanup
        inputTensor.close();
        results.close();

        return detections;
    }

    // Preprocess OpenCV Mat
    private float[] preprocessMat(Mat frame) {
        // Resize frame to model input size
        Mat resized = new Mat();
        Imgproc.resize(frame, resized, new Size(inputWidth, inputHeight));

        // Convert BGR to RGB
        Mat rgb = new Mat();
        Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);

        // Convert to float array in CHW format
        float[] data = new float[3 * inputHeight * inputWidth];
        int idx = 0;

        byte[] pixels = new byte[(int) rgb.total() * rgb.channels()];
        rgb.get(0, 0, pixels);

        // Convert to CHW format and normalize
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < inputHeight; h++) {
                for (int w = 0; w < inputWidth; w++) {
                    int pixelIndex = (h * inputWidth + w) * 3 + c;
                    data[idx++] = (pixels[pixelIndex] & 0xFF) / 255.0f;
                }
            }
        }

        return data;
    }
}