/******************************************************************************
 * Project Name:    [Gumball - VF17Detection]
 * Course:          [CECS1011 - Intro to CECS]
 * Semester:        [Fall 2025]
 * <p>
 * Members:         Dinh Hieu Minh <25minh.dh2@vinuni.edu.vn>,
 *                  Duc Phat Hoang <25phat.hd@vinuni.edu.vn>,
 *                  Le Ngoc Han <25han.ln@vinuni.edu.vn>,
 *                  Ngo Van Thang <25thang.nv@vinuni.edu.vn>,
 *                  Mai Dinh Vinh <25vinh.md@vinuni.edu.vn>
 * <p>
 * Date Created:    [10-15-2025]
 * Last Modified:   [10-22-2025]
 * <p>
 * File Name:       [DetectionHandler.java]
 * Developer:       Duc Phat Hoang, Ngo Van Thang
 * Description:     [The source code that handle image processing,
 *                  draw detection ring, crossing-line for arduino handling, opencv camera]
 ******************************************************************************/

package com.arthroverse.vf17.detection;

import ai.onnxruntime.OrtException;
import com.arthroverse.vf17.microcontroller.ArduinoComm;
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

/**
 * Handles real-time camera capture and YOLOv8 inference for VF17 detection.
 * It draws detection overlays, manages a virtual trigger line, and sends serial
 * signals to Arduino based on detected object classes and line-crossing logic.
 *
 * @author Duc Phat Hoang (25phat.hd@vinuni.edu.vn),
 *         Ngo Van Thang (25thang.nv@vinuni.edu.vn),
 *         Mai Dinh Vinh (25vinh.md@vinuni.edu.vn)
 * @version 1.0
 */
public class DetectionHandler {

    /**
     * All classes (labels) that are used to train the modeel
     */
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

    /**
     * These are all fields that are used to control the arduino serial signal sending by
     * create a visible line on the main UI
     */
    private static final int VIRTUAL_LINE_X = 500;

    //TODO: vinh
    private static final float TRIGGER_FRACTION = 0.5f;
    private enum MotionDirection { LEFT_TO_RIGHT, RIGHT_TO_LEFT }
    private static final MotionDirection MOTION_DIRECTION = MotionDirection.LEFT_TO_RIGHT;

    private final Map<String, Boolean> hasPassed = new HashMap<>();
    private final Set<String> alreadyTriggered = new HashSet<>();
    private static final long COOLDOWN_MS = 3000;

    private YOLOv8Detector detector; //main model inference instance
    private VideoCapture camera;    //main OpenCV camemra instance

    private final AtomicReference<BufferedImage> latestFrame = new AtomicReference<>();
    private final AtomicReference<List<YOLOv8Detector.Detection>> latestDetections = new AtomicReference<>();

    private final ExecutorService inferenceExecutor = Executors.newSingleThreadExecutor();
    private final ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();

    /**
     * Other variables for inference instance and OpenCV configurations
     */
    private volatile boolean isRunning = false;
    private volatile boolean isInferenceRunning = false;

    //camera frame FPS - Phat
    private int frameSkip = 2;
    private int frameCount = 0;
    private double currentFps = 0;

    /**
     * The constructor for the class where we load the OpenCV model on local machine and load the actual
     * ONNX runtime model file to OpenCV
     *
     * @author Ngo Van Thang (25thang.nv@vinuni.edu.vn)
     * @version 1.0
     * @throws OrtException if the ONNX Runtime environment or session cannot be initialized
     */
    public DetectionHandler() throws OrtException {
        nu.pattern.OpenCV.loadLocally();
        detector = new YOLOv8Detector("src/main/resources/model/best.onnx");
    }

    /**
     * This OpenCV camera activation syntax is the same as the Python syntax with the
     * <blockquote><pre>
     *     camera = new VideoCapture(0);
     * </pre></blockquote><p>
     * This is the python syntax
     * <blockquote><pre>
     *     import cv2
     *     cap = cv.VideoCapture(0);
     * </pre></blockquote><p>
     * There are 2 lines for camera resolution changes and 1 line to activate a separate thread
     * service for the OpenCV camera to prevent thread issue
     *
     * @author Duc Phat Hoang (25phat.hd@vinuni.edu.vn)
     * @version 1.0
     */
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

    /**
     * This method will be used to cleanup all resources related to the OpenCV camera such as the thread service,
     * the remaining frames of the BufferedFrame instances
     *
     * @author Duc Phat Hoang (25phat.hd@vinuni.edu.vn)
     * @version 1.0
     */
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

    /**
     * This method merely serves as a method to get the latest frame from the camera
     *
     * @author Duc Phat Hoang (25phat.hd@vinuni.edu.vn)
     * @version 1.0
     * @return the most recent camera frame as a {@link BufferedImage}
     */
    public BufferedImage getLatestFrame() {
        return latestFrame.get();
    }

    /**
     * Computes whether a detection has passed the virtual trigger line, based on a configurable
     * trigger point inside the bounding box and the selected motion direction.
     *
     * @param det the detection whose position is used for line-crossing evaluation
     * @return {@code true} if the detection has passed the trigger line based on the configured direction; otherwise {@code false}
     *
     * @author Dinh Hieu Minh (25minh.dh2@vinuni.edu.vn)
     * @version 1.0
     */
    private boolean hasPassed(YOLOv8Detector.Detection det) {
        float width = det.x2 - det.x1;
        if (width <= 0) {
            return false;
        }
        float triggerX = det.x1 + TRIGGER_FRACTION * width;
        if (MOTION_DIRECTION == MotionDirection.LEFT_TO_RIGHT) {
            return triggerX >= VIRTUAL_LINE_X;
        } else {
            return triggerX <= VIRTUAL_LINE_X;
        }
    }

    /**
     * This method will determine whether to trigger the output and also send a serial signal
     * to the Arduino board to activate the servo pusing<p>
     *
     * Each object will have a trigger id (based on their class name). If that id is inside the set of
     * all previously triggered ids, then they will be ignored => method returns {@code false}<p>
     *
     * Otherwise the method will trigger a 3000ms cooldown in the main Thread and
     * return {@code true} and then remove the triggeredid out of the set
     *
     * @author Ngo Van Thang (25thang.nv@vinuni.edu.vn)
     * @version 1.0
     *
     * bo qua
     */
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

    /**
     * This method will draw a visible virtual line on the main ui, indicating the border where
     * if an object passed, it will trigger the code and depend on the detection classes triggers a
     * serial signal to the Arduino board
     *
     * @author Ngo Van Thang (25thang.nv@vinuni.edu.vn)
     * @version 1.0
     */
    private void drawVirtualLine(Mat frame) {
        Scalar lineColor = new Scalar(255, 0, 0);
        int thickness = 3;
        Point start = new Point(VIRTUAL_LINE_X, 0);
        Point end = new Point(VIRTUAL_LINE_X, frame.rows());
        Imgproc.line(frame, start, end, lineColor, thickness);
    }

    /**
     * This is where all detection, fps count, conditional-based serial signal sending mechanism are implemented
     * This is where the real-time detection (drawing detection box) happens
     *
     * @author Duc Phat Hoang (25phat.hd@vinuni.edu.vn),
     *         Mai Dinh Vinh (25vinh.md@vinuni.edu.vn)
     * @version 1.0
     */
    private void runDetectionLoop() {
        Mat currentFrame = new Mat();
        Mat displayFrame = new Mat();
        long lastTime = System.currentTimeMillis();
        int fpsFrameCount = 0;
        try {
            while (isRunning && camera != null && camera.isOpened()) {
                boolean success = camera.read(currentFrame);
                if (!isRunning || !success || currentFrame.empty()) {
                    break;
                }
                frameCount++;
                currentFrame.copyTo(displayFrame);
                if (frameCount % frameSkip == 0 && !isInferenceRunning) {
                    Mat frameForInference = currentFrame.clone();
                    isInferenceRunning = true;
                    inferenceExecutor.submit(() -> {
                        try {
                            List<YOLOv8Detector.Detection> detections = detector.detect(frameForInference);
                            latestDetections.set(detections);
                            if (!detections.isEmpty()) {
                                for (YOLOv8Detector.Detection det : detections) {
                                    String className = ALL_CLASSES[det.classId];
                                    boolean objectHasPassed = hasPassed(det);

                                    if (shouldTriggerOutput(className, objectHasPassed)) {
                                        String inferOutput = "Class: %s, Confidence: %.2f"
                                                .formatted(className, det.confidence);
                                        HomepageUIController.frontendUpdateOutput(
                                                inferOutput,
                                                className.contains("rotten")
                                        );
                                        ArduinoComm.COMMUNICATE(className.contains("rotten"));
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

    /**
     * This method is responsible to draw the detection box based on the classification
     * This method will draw a rectangle with specified color, classification text and
     * inference confidence in real time<p>
     *
     * This method will recieve an image frame in Matrix format and process it
     *
     * @author Duc Phat Hoang (25phat.hd@vinuni.edu.vn)
     * @version 1.0
     */
    private void drawDetections(Mat frame, List<YOLOv8Detector.Detection> detections) {
        for (YOLOv8Detector.Detection det : detections) {
            Point topLeft = new Point(det.x1, det.y1);
            Point bottomRight = new Point(det.x2, det.y2);
            Scalar color;
            if (ALL_CLASSES[det.classId].contains("rotten")) {
                color = new Scalar(0, 0, 255);
            } else {
                color = new Scalar(0, 255, 0);
            }
            Imgproc.rectangle(frame, topLeft, bottomRight, color, 2);
            String label = String.format("Class: %s: %.2f",
                    ALL_CLASSES[det.classId], det.confidence);
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

    /**
     * This method is used to convert a frame matrix (including the drawn detection box)
     * to a BufferedImage
     *
     * @author Ngo Van Thang (25thang.nv@vinuni.edu.vn)
     * @version 1.0
     */
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

    /**
     * This method will cleanup all of the processes related to the program such as inference instance,
     * thread service for the inference instance, close the camera
     *
     * @author Ngo Van Thang (25thang.nv@vinuni.edu.vn)
     * @version 1.0
     */
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

    /**
     * This method is used to shutdown the whole program including all resources + the camera and
     * remove the YOLODetection inference instance and then close it properly
     *
     * @author Duc Phat Hoang (25phat.hd@vinuni.edu.vn)
     * @version 1.0
     */
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
