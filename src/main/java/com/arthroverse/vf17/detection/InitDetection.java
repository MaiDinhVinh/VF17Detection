package com.arthroverse.vf17.detection;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;

public class InitDetection {

    private static final String[] COCO_CLASSES = {
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
    private JFrame frame;
    private JLabel imageLabel;
    private AtomicReference<List<YOLOv8Detector.Detection>> latestDetections =
            new AtomicReference<>();
    private ExecutorService inferenceExecutor = Executors.newSingleThreadExecutor();
    private volatile boolean isInferenceRunning = false;
    private int frameSkip = 2; // Process every 2nd frame
    private int frameCount = 0;

    public InitDetection() throws Exception {
        // Load OpenCV native library
        nu.pattern.OpenCV.loadLocally();

        // Load YOLOv8 model

        detector = new YOLOv8Detector("src/main/resources/model/best.onnx");
        System.out.println("model loaded");

        // Initialize display window
        frame = new JFrame("YOLOv8 Detection");
        imageLabel = new JLabel();
        frame.add(imageLabel);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(640, 480);
        frame.setVisible(true);
    }

    public void startWebcam() {
        camera = new VideoCapture(0); // 0 for default webcam

        if (!camera.isOpened()) {
            System.err.println("Error: Cannot open webcam!");
            return;
        }

        System.out.println("Webcam opened. Press 'q' to quit.");
        runDetection();
    }


    private void runDetection() {
        Mat currentFrame = new Mat();
        Mat displayFrame = new Mat();
        long lastTime = System.currentTimeMillis();
        int fpsFrameCount = 0;
        double currentFps = 0;

        while (camera.read(currentFrame)) {
            if (currentFrame.empty()) {
                break;
            }

            frameCount++;

            // Clone frame for display (don't wait for inference)
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
                            System.out.println("\n========== Frame " + frameCount + " ==========");
                            System.out.println("Found " + detections.size() + " object(s):");
                            for (int i = 0; i < detections.size(); i++) {
                                YOLOv8Detector.Detection det = detections.get(i);
                                System.out.printf("[%d] Class: %s, Confidence: %.2f, Box: [%.1f, %.1f, %.1f, %.1f]\n",
                                        i + 1,
                                        COCO_CLASSES[det.classId],
                                        det.confidence,
                                        det.x1, det.y1, det.x2, det.y2);
                            }
                            System.out.println("=====================================\n");
                        }

                    } catch (Exception e) {
                        e.printStackTrace();
                    } finally {
                        frameForInference.release();
                        isInferenceRunning = false;
                    }
                });
            }

            // Draw latest detections (even if from previous frame)
            List<YOLOv8Detector.Detection> detections = latestDetections.get();
            if (detections != null) {
                drawDetections(displayFrame, detections);
            }

            // Calculate and display FPS
            fpsFrameCount++;
            long currentTime = System.currentTimeMillis();
            if (currentTime - lastTime >= 1000) {
                currentFps = fpsFrameCount * 1000.0 / (currentTime - lastTime);
                fpsFrameCount = 0;
                lastTime = currentTime;
            }

            String fpsText = String.format("FPS: %.1f | Inference: %s",
                    currentFps, isInferenceRunning ? "Running" : "Ready");
            Imgproc.putText(displayFrame, fpsText, new Point(10, 30),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 0), 2);

            // Display frame immediately
            displayFrame(displayFrame);

            // Small delay to prevent CPU overload
            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                break;
            }
        }

        cleanup();
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
                    COCO_CLASSES[det.classId], det.confidence);

            int baseline[] = {0};
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

    private void displayFrame(Mat frame) {
        BufferedImage image = matToBufferedImage(frame);
        ImageIcon icon = new ImageIcon(image);
        imageLabel.setIcon(icon);
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
        if (camera != null) {
            camera.release();
        }
        try {
            if (detector != null) {
                detector.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        frame.dispose();
    }

    /**
     * Print detection results to console
     */
    public void printDetections(Mat frame) {
        try {
            List<YOLOv8Detector.Detection> detections = detector.detect(frame);

            System.out.println("\n--- Detections ---");
            for (YOLOv8Detector.Detection det : detections) {
                System.out.printf("Class: %s, Confidence: %.2f, Box: [%.1f, %.1f, %.1f, %.1f]\n",
                        COCO_CLASSES[det.classId],
                        det.confidence,
                        det.x1, det.y1, det.x2, det.y2);
            }
            System.out.println("------------------\n");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}