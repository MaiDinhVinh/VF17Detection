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
 * File Name:       [HomepageUIController.java]
 * Developer:       Le Ngoc Han
 * Description:     [JavaFX UI controller, controlling what happen on the frontend]
 ******************************************************************************/

package com.arthroverse.vf17.uicontrollers;

import com.arthroverse.vf17.detection.DetectionHandler;

import io.github.palexdev.materialfx.controls.MFXScrollPane;
import javafx.animation.AnimationTimer;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;


import java.awt.image.BufferedImage;
import java.net.URL;
import java.util.ResourceBundle;

public class HomepageUIController implements Initializable {

    @FXML
    private ImageView mainCamView;

    @FXML
    private VBox outputLogContainer;

    @FXML
    private MFXScrollPane inferOutputPane;

    private DetectionHandler detectionHandler;

    private AnimationTimer frameUpdater;

    private static VBox outputLogContainerStatic;

    private static MFXScrollPane inferOutputPaneStatic;

    private static int MAX_RECORD = 10;

    /**
     * This method will set everything up like the ImageView integration with
     * Java AWT Buffered Image so that it acts like a video camera view, initalize the
     * detection handler, or anything else you need to setup before initialize the UI
     *
     * @author Le Ngoc Han
     *
     * @Version 1.0
     */
    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        try {
            outputLogContainerStatic = outputLogContainer;
            inferOutputPaneStatic = inferOutputPane;
            DetectionHandler Handler = new DetectionHandler();
            detectionHandler = Handler;
            setupFrameUpdater();

            mainCamView.setPreserveRatio(true);
            mainCamView.setSmooth(true);

            detectionHandler.startCamera();
            frameUpdater.start();

            BufferedImage bufferedImage = detectionHandler.getLatestFrame();

            if (bufferedImage != null) {
                Image fxImage = SwingFXUtils.toFXImage(bufferedImage, null);
                mainCamView.setImage(fxImage);
            }

            frameUpdater = new AnimationTimer() {
                @Override
                public void handle(long now) {
                    updateFrame();
                }
            };
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * This method sets an animation timer which will assign a new frame to the ImageView
     * and acts like video camera frame
     * @author Le Ngoc Han
     *
     * @Version 1.0
     */
    private void setupFrameUpdater() {
        frameUpdater = new AnimationTimer() {
            @Override
            public void handle(long now) {
                updateFrame();
            }
        };
    }

    /**
     * This method will get the latest frame from OpenCV camera implementation and
     * assign it to the ImageView
     * @author Le Ngoc Han
     *
     * @Version 1.0
     */
    private void updateFrame() {
        BufferedImage bufferedImage = detectionHandler.getLatestFrame();

        if (bufferedImage != null) {
            Image fxImage = SwingFXUtils.toFXImage(bufferedImage, null);
            mainCamView.setImage(fxImage);
        }
    }

    /**
    * This method will cleanup all resources such as camera and YOLO Inference objects
    * such as Concurrency objects,...
    * @author Le Ngoc Han
    *
    * @Version 1.0
    * */
    public void shutdown() {
        if (frameUpdater != null) {
            frameUpdater.stop();
        }
        if (detectionHandler != null) {
            detectionHandler.shutdown();
        }
    }

    /**
     * This method will assign a listener to the current UI and will initiate
     * the shutdown whenever the user close the windows
     * @author Le Ngoc Han
     *
     * @Version 1.0
     */
    public void handleShutdown(Stage stage){
        stage.setOnCloseRequest(event -> {
            shutdown();
        });
    }

    /**
     * This method will add inference output to a separate VBox for more information to the user
     * This method not only add the inference output but also add some CSS effects to the text
     * like fonts, size, color based on the output
     *
     * @author Le Ngoc Han
     *
     * @Version 1.0
     */
    public static void frontendUpdateOutput(String inferOutput, boolean isRotten){
        Platform.runLater(() -> {
            Label label = new Label(inferOutput);
            if(isRotten){
                label.setStyle("-fx-font-size: 16px; -fx-text-fill: red; -fx-font-family: 'JetBrains Mono Regular';");
            }else{
                label.setStyle("-fx-font-size: 16px; -fx-text-fill: green; -fx-font-family: 'JetBrains Mono Regular';");
            }
            label.setWrapText(true);
            outputLogContainerStatic.getChildren().add(label);
            inferOutputPaneStatic.setVvalue(1.0);
            if(outputLogContainerStatic.getChildren().size() > MAX_RECORD){
                outputLogContainerStatic.getChildren().remove(0, outputLogContainerStatic.getChildren().size() - 1 - 10);
            }
        });
    }
}
