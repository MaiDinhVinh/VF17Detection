package com.arthroverse.vf17.main;

import com.arthroverse.vf17.detection.InitDetection;

public class VF17Main {
    public static void main(String[] args) {
        try {
            InitDetection init = new InitDetection();

            // For webcam
            init.startWebcam();

            // For video file
            // init.startVideo("path/to/your_video.mp4");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
