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
 * File Name:       [YOLOv8Detector.java]
 * Developer:       Duc Phat Hoang, Ngo Van Thang, Mai Dinh Vinh
 * Description:     [The source code that is the main Java Classifications model]
 ******************************************************************************/
package com.arthroverse.vf17.detection;

import ai.onnxruntime.*;
import java.nio.FloatBuffer;
import java.util.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class YOLOv8Detector {
    /**
     * There are all of the fields (parameters) used for the main YOLO Classification models
     *
     * @author Mai Dinh Vinh
     *
     * @Version 1.0
     * */
    private OrtEnvironment env; //this is the ONNX Runtime Environment
    private OrtSession session; //This is the ONNX session
    private final int inputWidth = 640; //the width of the input image
    private final int inputHeight = 640; //the height of the input image

    /**
     * This is called the Confidence Threshold, where certain objects with certain detection confidence
     * will be keep. In this situation, all objects with detection confidence >= 0.25*/
    private final float confThreshold = 0.25f;

    /**
     * This is called Intersection over Union threshold<p>
     *
     * A same object might be detected twice or more with slightly different boxes.
     * Therefore, Interfection over Union threshold is used to determine the amount of overlap between
     * the detection boxes and if they are in the same classes => keep the highest confidence box*/
    private final float iouThreshold = 0.45f;

    /**
     * This constructor will initalize the ONNX Runtime environment and ONNX session and then
     * pass the model path to the method {@code createSession(modelPaths ,opts}
     *
     * @author Mai Dinh Vinh
     *
     * @Version 1.0
     * */
    public YOLOv8Detector(String modelPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        session = env.createSession(modelPath, opts);
    }

    /**
     * This method will return a transposed matrix from the ONNX Runtime result
     *
     * @author Mai Dinh Vinh
     *
     * @Version 1.0
     * */
    private float[][] processOutput(OrtSession.Result results) throws OrtException {
        OnnxValue outputValue = results.get(0);
        float[][][] rawOutput = (float[][][]) outputValue.getValue();

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

    /**
     * This method will return a list consisting of Detection boxes which has been created
     * after the process of raw output from the ONNX runtime<p>
     *
     * This method will determine if the confidence is strictly higher than the {@code confThreshold}
     * and then create a new Detection box instance and add it to the returning list
     *
     * @author Mai Dinh Vinh
     *
     * @Version 1.0
     * */
    private List<Detection> postProcess(float[][] output, int originalWidth,
                                        int originalHeight) {
        List<Detection> detections = new ArrayList<>();

        for (float[] detection : output) {
            float x_center = detection[0];
            float y_center = detection[1];
            float width = detection[2];
            float height = detection[3];

            float maxConf = 0;
            int classId = 0;
            for (int i = 4; i < detection.length; i++) {
                if (detection[i] > maxConf) {
                    maxConf = detection[i];
                    classId = i - 4;
                }
            }

            if (maxConf > confThreshold) {
                float x1 = (x_center - width / 2) * originalWidth / inputWidth;
                float y1 = (y_center - height / 2) * originalHeight / inputHeight;
                float x2 = (x_center + width / 2) * originalWidth / inputWidth;
                float y2 = (y_center + height / 2) * originalHeight / inputHeight;

                detections.add(new Detection(x1, y1, x2, y2, maxConf, classId));
            }
        }

        return applyNMS(detections);
    }

    /**
     * This method will be used to apply the Intersection over Union threshold mechanism
     * commonly known as (Non-maximum Suppression). After determined which detection boxes has the
     * highest confidence, NMS will be used to eliminate duplication boxes based on the
     * iOu threshold declared above
     *
     * @author Ngo Van Thang
     *
     * @Version 1.0
     * */
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

    /**
     * This is a helper method for the {@code applyNMS(List<Detection> detections)} method above
     * where it calculates the IoU value between 2 nearest detections
     *
     * @author Ngo Van Thang
     *
     * @Version 1.0*/
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

    /**
     * This method is used to close all ONNX runtime environment and the
     * ONNX runtime
     *
     * @author Ngo Van Thang
     *
     * @Version 1.0
     * */
    public void close() throws OrtException {
        session.close();
        env.close();
    }

    /**
     * This class serves as a detection box instance
     *
     * @author Duc Phat Hoang
     *
     * @Version 1.0
     * */
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

    /**
    * This method will recieve an image frame as a matrix and then apply all calculation above
    * and produce a list of all detection boxes with mathematical formulas applied
    *
    * @author Duc Phat Hoang
    *
    * @Version 1.0
    * */
    public List<Detection> detect(Mat frame) throws OrtException {
        float[] inputData = preprocessMat(frame);

        long[] shape = {1, 3, inputHeight, inputWidth};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env,
                FloatBuffer.wrap(inputData), shape);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(session.getInputNames().iterator().next(), inputTensor);

        OrtSession.Result results = session.run(inputs);

        float[][] output = processOutput(results);
        List<Detection> detections = postProcess(output, frame.width(), frame.height());

        inputTensor.close();
        results.close();

        return detections;
    }

    /**
     * This method will recieve a raw image frame as a matrix and then process it into a
     * float array of rgb color for further processing and calculation to produce a final
     * classification
     *
     * @author Duc Phat Hoang
     *
     * @Version 1.0
     * */
    private float[] preprocessMat(Mat frame) {
        Mat resized = new Mat();
        Imgproc.resize(frame, resized, new Size(inputWidth, inputHeight));

        Mat rgb = new Mat();
        Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);

        float[] data = new float[3 * inputHeight * inputWidth];
        int idx = 0;

        byte[] pixels = new byte[(int) rgb.total() * rgb.channels()];
        rgb.get(0, 0, pixels);

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