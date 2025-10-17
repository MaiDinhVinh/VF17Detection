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

    // Virtual line configuration
    private static final int VIRTUAL_LINE_X = 1200; // Vertical line at center (860/2)
    private static final boolean USE_VERTICAL_LINE = true; // true for vertical, false for horizontal
    private static final int VIRTUAL_LINE_Y = 287; // Horizontal line at center (574/2)

    // Tracking which objects have passed the line
    private final Map<String, Boolean> hasPassed = new HashMap<>();
    private final Set<String> alreadyTriggered = new HashSet<>();
    private static final long COOLDOWN_MS = 3000; // 3 second cooldown before same class can trigger again

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

    private boolean hasPassed(YOLOv8Detector.Detection det) {
        if (USE_VERTICAL_LINE) {
            // Object has passed if its center is to the left of the line
            float centerX = (det.x1 + det.x2) / 2;
            return centerX < VIRTUAL_LINE_X;
        } else {
            // Object has passed if its center is above the line
            float centerY = (det.y1 + det.y2) / 2;
            return centerY < VIRTUAL_LINE_Y;
        }
    }

    private boolean shouldTriggerOutput(String className, boolean currentlyPassed) {
        String triggerId = className;

        if (alreadyTriggered.contains(triggerId)) {
            return false;
        }

        Boolean previouslyPassed = hasPassed.get(className);

        hasPassed.put(className, currentlyPassed);

        if (previouslyPassed != null && !previouslyPassed && currentlyPassed) {
            alreadyTriggered.add(triggerId);

            new Thread(() -> {
                try {
                    Thread.sleep(COOLDOWN_MS);
                    alreadyTriggered.remove(triggerId);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }).start();

            return true;
        }

        return false;
    }

    private void drawVirtualLine(Mat frame) {
        Scalar lineColor = new Scalar(255, 0, 0); // Red line
        int thickness = 3;

        if (USE_VERTICAL_LINE) {
            // Draw vertical line
            Point start = new Point(VIRTUAL_LINE_X, 0);
            Point end = new Point(VIRTUAL_LINE_X, frame.rows());
            Imgproc.line(frame, start, end, lineColor, thickness);
        } else {
            Point start = new Point(0, VIRTUAL_LINE_Y);
            Point end = new Point(frame.cols(), VIRTUAL_LINE_Y);
            Imgproc.line(frame, start, end, lineColor, thickness);
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

                            if (!detections.isEmpty()) {
                                for (YOLOv8Detector.Detection det : detections) {
                                    String className = ALL_CLASSES[det.classId];
                                    boolean objectHasPassed = hasPassed(det);

                                    // Check if object is transitioning from right to left (crossing the line)
                                    if (shouldTriggerOutput(className, objectHasPassed)) {
                                        String inferOutput = "✓ PASSED: %s, Confidence: %.2f"
                                                .formatted(className, det.confidence);
                                        HomepageUIController.frontendUpdateOutput(
                                                inferOutput,
                                                className.contains("rotten")
                                        );
                                    }
                                }
                            }
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

                drawVirtualLine(displayFrame);

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

            Scalar color;
            String status;
            if (hasPassed(det)) {
                color = new Scalar(0, 0, 255);
                status = "PASSED";
            } else {
                color = new Scalar(0, 255, 0);
                status = "NOT PASSED";
            }

            Imgproc.rectangle(frame, topLeft, bottomRight, color, 2);

            String label = String.format("%s: %.2f [%s]",
                    ALL_CLASSES[det.classId], det.confidence, status);
            int[] baseline = {0};
            Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.5, 1, baseline);

            Point labelOrigin = new Point(det.x1, det.y1 - 10);
            Imgproc.rectangle(frame,
                    new Point(det.x1, det.y1 - labelSize.height - 10),
                    new Point(det.x1 + labelSize.width, det.y1),
                    color, -1);

            Imgproc.putText(frame, label, labelOrigin,
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 0), 1);
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