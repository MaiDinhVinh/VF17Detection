package com.arthroverse.vf17.uicontrollers;

import com.arthroverse.vf17.detection.DetectionHandler;

import io.github.palexdev.materialfx.controls.MFXScrollPane;
import javafx.animation.AnimationTimer;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
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

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        try {
            outputLogContainerStatic = outputLogContainer;
            inferOutputPaneStatic = inferOutputPane;
            DetectionHandler Handler = new DetectionHandler();
            detectionHandler = Handler;
            setupFrameUpdater();

            // Configure ImageView for smooth display
            mainCamView.setPreserveRatio(true);
            mainCamView.setSmooth(true);

            detectionHandler.startCamera();
            frameUpdater.start();

            BufferedImage bufferedImage = detectionHandler.getLatestFrame();

            if (bufferedImage != null) {
                // Convert BufferedImage to JavaFX Image
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

    private void setupFrameUpdater() {
        frameUpdater = new AnimationTimer() {
            @Override
            public void handle(long now) {
                updateFrame();
            }
        };
    }

    private void updateFrame() {
        BufferedImage bufferedImage = detectionHandler.getLatestFrame();

        if (bufferedImage != null) {
            // Convert BufferedImage to JavaFX Image
            Image fxImage = SwingFXUtils.toFXImage(bufferedImage, null);
            mainCamView.setImage(fxImage);
        }
    }

    public void shutdown() {
        if (frameUpdater != null) {
            frameUpdater.stop();
        }
        if (detectionHandler != null) {
            detectionHandler.shutdown();
        }
    }

    public void handleShutdown(Stage stage){
        stage.setOnCloseRequest(event -> {
            shutdown();
        });
    }

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
