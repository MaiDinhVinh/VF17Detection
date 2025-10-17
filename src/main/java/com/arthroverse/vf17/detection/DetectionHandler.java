package com.arthroverse.vf17.detection;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import com.arthroverse.vf17.uicontrollers.HomepageUIController;

import com.fazecast.jSerialComm.SerialPort;

public class DetectionHandler {

    private static final String[] ALL_CLASSES = {
            "fresh apple",
            "fresh banana",
            "fresh bellpepper",
            "fresh carrot",
            "fresh cucumber",
            "fresh mango",
            "fresh orange",
            "fresh potato",
            "rotten apple",
            "rotten banana",
            "rotten carrot",
            "rotten cucumber",
            "rotten mango",
            "rotten orange",
            "rotten potato",
            "rotten tomato",
            "rottenbellpepper"
    };

    private YOLOv8Detector detector;
    private VideoCapture camera;
    private AtomicReference<BufferedImage> latestFrame = new AtomicReference<>();
    private AtomicReference<List<YOLOv8Detector.Detection>> latestDetections = new AtomicReference<>();
    private ExecutorService inferenceExecutor = Executors.newSingleThreadExecutor();
    private ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();
    private volatile boolean isRunning = false;
    private volatile boolean isInferenceRunning = false;
    private int frameSkip = 2;
    private int frameCount = 0;
    private double currentFps = 0;

    private final Map<String, Long> detectedObjects = new HashMap<>();
    private static final int CONSECUTIVE_FRAMES_REQUIRED = 30;
    private final Map<String, Integer> detectionCounter = new HashMap<>();

    public DetectionHandler() throws Exception {
        nu.pattern.OpenCV.loadLocally();

        detector = new YOLOv8Detector("src/main/resources/model/best.onnx");
    }

    public void startCamera() {
        if (isRunning) {
            return;
        }
        camera = new VideoCapture(0);

        if (!camera.isOpened()) {
            return;
        }

        camera.set(Videoio.CAP_PROP_FRAME_WIDTH, 860);
        camera.set(Videoio.CAP_PROP_FRAME_HEIGHT, 574);

        isRunning = true;

        cameraExecutor.submit(this::runDetectionLoop);
    }


    public void stopCamera() {
        isRunning = false;
        try {
            cameraExecutor.shutdown();
            if (!cameraExecutor.awaitTermination(2, TimeUnit.SECONDS)) {
                cameraExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            cameraExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        cleanup();
    }

    public BufferedImage getLatestFrame() {
        return latestFrame.get();
    }

    public List<YOLOv8Detector.Detection> getLatestDetections() {
        return latestDetections.get();
    }

    public double getCurrentFps() {
        return currentFps;
    }

    public boolean isRunning() {
        return isRunning;
    }

    public static String getClassName(int classId) {
        if (classId >= 0 && classId < ALL_CLASSES.length) {
            return ALL_CLASSES[classId];
        }
        return "Unknown";
    }

    private boolean shouldReportDetection(String className) {
        long currentTime = System.currentTimeMillis();
        detectionCounter.put(className, detectionCounter.getOrDefault(className, 0) + 1);
        if (detectionCounter.get(className) >= CONSECUTIVE_FRAMES_REQUIRED) {
            detectedObjects.put(className, currentTime);
            detectionCounter.put(className, 0);
            return true;
        }
        return false;
    }

    private void resetUndetectedObjects(Set<String> currentlyDetectedClasses) {
        Set<String> allTrackedClasses = new HashSet<>(detectionCounter.keySet());
        for (String className : allTrackedClasses) {
            if (!currentlyDetectedClasses.contains(className)) {
                detectionCounter.put(className, 0);
            }
        }
    }

    private void runDetectionLoop() {
        Mat currentFrame = new Mat();
        Mat displayFrame = new Mat();
        long lastTime = System.currentTimeMillis();
        int fpsFrameCount = 0;
        try {
            while (isRunning && camera != null && camera.isOpened()) {
                if (!isRunning) {
                    break;
                }

                boolean success = camera.read(currentFrame);

                if (!success || currentFrame.empty()) {
                    break;
                }

                frameCount++;

                currentFrame.copyTo(displayFrame);

                if (frameCount % frameSkip == 0 && !isInferenceRunning) {
                    Mat frameForInference = currentFrame.clone();
                    isInferenceRunning = true;
                    inferenceExecutor.submit(() -> {
                        try {
                            List<YOLOv8Detector.Detection> detections =
                                    detector.detect(frameForInference);
                            latestDetections.set(detections);
                            Set<String> currentlyDetectedClasses = new HashSet<>();
                            if (!detections.isEmpty()) {
                                for (YOLOv8Detector.Detection det : detections) {
                                    String className = ALL_CLASSES[det.classId];
                                    currentlyDetectedClasses.add(className);
                                    if (shouldReportDetection(className)) {
                                        String inferOutput = "Class: %s, Confidence: %.2f"
                                                .formatted(className, det.confidence);
                                        HomepageUIController.frontendUpdateOutput(inferOutput, className.contains("rotten"));
                                    }
                                }
                            }
                            resetUndetectedObjects(currentlyDetectedClasses);
                        } catch (Exception e) {
                            e.printStackTrace();
                        } finally {
                            frameForInference.release();
                            isInferenceRunning = false;
                        }
                    });
                }
                List<YOLOv8Detector.Detection> detections = latestDetections.get();
                if (detections != null) {
                    drawDetections(displayFrame, detections);
                }
                fpsFrameCount++;
                long currentTime = System.currentTimeMillis();
                if (currentTime - lastTime >= 1000) {
                    currentFps = fpsFrameCount * 1000.0 / (currentTime - lastTime);
                    fpsFrameCount = 0;
                    lastTime = currentTime;
                }
                String fpsText = String.format("FPS: %.1f", currentFps);
                Imgproc.putText(displayFrame, fpsText, new Point(10, 30),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 0), 2);

                BufferedImage bufferedImage = matToBufferedImage(displayFrame);
                latestFrame.set(bufferedImage);
                try {
                    Thread.sleep(1);
                } catch (InterruptedException e) {
                    break;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            currentFrame.release();
            displayFrame.release();
        }
    }

    private void drawDetections(Mat frame, List<YOLOv8Detector.Detection> detections) {
        for (YOLOv8Detector.Detection det : detections) {
            Point topLeft = new Point(det.x1, det.y1);
            Point bottomRight = new Point(det.x2, det.y2);
            Scalar color = new Scalar(0, 255, 0); // Green
            Imgproc.rectangle(frame, topLeft, bottomRight, color, 2);

            String label = String.format("%s: %.2f",
                    ALL_CLASSES[det.classId], det.confidence);
            int[] baseline = {0};
            Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX,
                    2.0, 2, baseline);

            Point labelOrigin = new Point(det.x1, det.y1 - 10);
            Imgproc.rectangle(frame,
                    new Point(det.x1, det.y1 - labelSize.height - 10),
                    new Point(det.x1 + labelSize.width, det.y1),
                    color, -1);

            Imgproc.putText(frame, label, labelOrigin,
                    Imgproc.FONT_HERSHEY_SIMPLEX, 2.0, new Scalar(0, 0, 0), 2);
        }
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        mat.get(0, 0, data);
        return image;
    }

    private void cleanup() {
        if (camera != null && camera.isOpened()) {
            camera.release();
            camera = null;
        }
        if (inferenceExecutor != null && !inferenceExecutor.isShutdown()) {
            inferenceExecutor.shutdown();
            try {
                if (!inferenceExecutor.awaitTermination(1, TimeUnit.SECONDS)) {
                    inferenceExecutor.shutdownNow();
                }
            } catch (InterruptedException e) {
                inferenceExecutor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    public void shutdown() {
        if (isRunning) {
            stopCamera();
        }
        try {
            if (detector != null) {
                detector.close();
                detector = null;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}