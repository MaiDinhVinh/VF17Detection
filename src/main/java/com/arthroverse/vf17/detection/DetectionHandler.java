package com.arthroverse.vf17.detection;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.List;
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
        // Load OpenCV native library
        nu.pattern.OpenCV.loadLocally();

        // Load YOLOv8 model
        detector = new YOLOv8Detector("src/main/resources/model/best.onnx");
        System.out.println("Model loaded successfully");
    }

    /**
     * Start the camera and detection loop
     */
    public void startCamera() {
        if (isRunning) {
            return;
        }

        camera = new VideoCapture(0);

        if (!camera.isOpened()) {
            return;
        }

        // Set camera resolution
        camera.set(Videoio.CAP_PROP_FRAME_WIDTH, 860);
        camera.set(Videoio.CAP_PROP_FRAME_HEIGHT, 574);

        isRunning = true;
        System.out.println("Camera started");

        // Run detection loop in background thread
        cameraExecutor.submit(this::runDetectionLoop);
    }

    /**a
     * Stop the camera and cleanup resources
     */
    public void stopCamera() {
        isRunning = false;

        // Wait for camera thread to finish
        try {
            cameraExecutor.shutdown();
            if (!cameraExecutor.awaitTermination(2, TimeUnit.SECONDS)) {
                cameraExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            cameraExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        // Now safe to cleanup camera
        cleanup();
    }

    /**
     * Get the latest processed frame with detections drawn
     * @return BufferedImage of the latest frame, or null if no frame available
     */
    public BufferedImage getLatestFrame() {
        return latestFrame.get();
    }

    /**
     * Get the latest detections
     * @return List of detections, or null if no detections available
     */
    public List<YOLOv8Detector.Detection> getLatestDetections() {
        return latestDetections.get();
    }

    /**
     * Get current FPS
     * @return Current frames per second
     */
    public double getCurrentFps() {
        return currentFps;
    }

    /**
     * Check if camera is running
     * @return true if camera is active
     */
    public boolean isRunning() {
        return isRunning;
    }

    /**
     * Get class name by index
     * @param classId Class index
     * @return Class name
     */
    public static String getClassName(int classId) {
        if (classId >= 0 && classId < ALL_CLASSES.length) {
            return ALL_CLASSES[classId];
        }
        return "Unknown";
    }

    private void runDetectionLoop() {
        Mat currentFrame = new Mat();
        Mat displayFrame = new Mat();
        long lastTime = System.currentTimeMillis();
        int fpsFrameCount = 0;

        try {
            while (isRunning && camera != null && camera.isOpened()) {
                // Check if we should stop BEFORE reading
                if (!isRunning) {
                    break;
                }

                boolean success = camera.read(currentFrame);

                if (!success || currentFrame.empty()) {
                    break;
                }

                frameCount++;

                // Clone frame for display
                currentFrame.copyTo(displayFrame);

                // Run inference on every Nth frame asynchronously
                if (frameCount % frameSkip == 0 && !isInferenceRunning) {
                    Mat frameForInference = currentFrame.clone();
                    isInferenceRunning = true;

                    inferenceExecutor.submit(() -> {
                        try {
                            List<YOLOv8Detector.Detection> detections =
                                    detector.detect(frameForInference);
                            latestDetections.set(detections);

                            // Print detections to console
                            if (!detections.isEmpty()) {
                                for (YOLOv8Detector.Detection det : detections) {
                                    System.out.printf("Class: %s, Confidence: %.2f, Box: [%.1f, %.1f, %.1f, %.1f]\n",
                                            ALL_CLASSES[det.classId],
                                            det.confidence,
                                            det.x1, det.y1, det.x2, det.y2);
                                    String inferOutput = "Class: %s, Confidence: %.2f"
                                            .formatted(ALL_CLASSES[det.classId], det.confidence);
                                    if(ALL_CLASSES[det.classId].contains("rotten")){
                                        HomepageUIController.frontendUpdateOutput(inferOutput, true);
                                    }else{
                                        HomepageUIController.frontendUpdateOutput(inferOutput, false);
                                    }
                                }
                            }

                        } catch (Exception e) {
                        } finally {
                            frameForInference.release();
                            isInferenceRunning = false;
                        }
                    });
                }

                // Draw latest detections
                List<YOLOv8Detector.Detection> detections = latestDetections.get();
                if (detections != null) {
                    drawDetections(displayFrame, detections);
                }

                // Calculate FPS
                fpsFrameCount++;
                long currentTime = System.currentTimeMillis();
                if (currentTime - lastTime >= 1000) {
                    currentFps = fpsFrameCount * 1000.0 / (currentTime - lastTime);
                    fpsFrameCount = 0;
                    lastTime = currentTime;
                }

                // Draw FPS on frame
                String fpsText = String.format("FPS: %.1f", currentFps);
                Imgproc.putText(displayFrame, fpsText, new Point(10, 30),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 0), 2);

                // Convert and store the frame
                BufferedImage bufferedImage = matToBufferedImage(displayFrame);
                latestFrame.set(bufferedImage);

                // Small delay
                try {
                    Thread.sleep(1);
                } catch (InterruptedException e) {
                    break;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // Clean up Mats
            currentFrame.release();
            displayFrame.release();
        }
    }

    private void drawDetections(Mat frame, List<YOLOv8Detector.Detection> detections) {
        for (YOLOv8Detector.Detection det : detections) {
            // Draw bounding box
            Point topLeft = new Point(det.x1, det.y1);
            Point bottomRight = new Point(det.x2, det.y2);
            Scalar color = new Scalar(0, 255, 0); // Green
            Imgproc.rectangle(frame, topLeft, bottomRight, color, 2);

            // Draw label with confidence
            String label = String.format("%s: %.2f",
                    ALL_CLASSES[det.classId], det.confidence);

            int[] baseline = {0};
            Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.5, 1, baseline);

            // Draw label background
            Point labelOrigin = new Point(det.x1, det.y1 - 10);
            Imgproc.rectangle(frame,
                    new Point(det.x1, det.y1 - labelSize.height - 10),
                    new Point(det.x1 + labelSize.width, det.y1),
                    color, -1);

            // Draw label text
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
        // Release camera first
        if (camera != null && camera.isOpened()) {
            camera.release();
            camera = null;
        }

        // Shutdown inference executor
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

    /**
     * Complete shutdown of all resources
     */
    public void shutdown() {

        // Stop camera first
        if (isRunning) {
            stopCamera();
        }

        // Close detector
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