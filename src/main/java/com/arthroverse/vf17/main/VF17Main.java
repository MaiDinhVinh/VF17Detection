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
 * Last Modified:   [20-1-2026]
 * <p>
 * File Name:       [VF17Main.java]
 * Developer:       Le Ngoc Han
 * Description:     [The main entry point of the entire application]
 ******************************************************************************/

package com.arthroverse.vf17.main;

import com.arthroverse.vf17.uicontrollers.HomepageUIController;
import io.github.palexdev.materialfx.theming.JavaFXThemes;
import io.github.palexdev.materialfx.theming.MaterialFXStylesheets;
import io.github.palexdev.materialfx.theming.UserAgentBuilder;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

/**
 * Main entry point of the VF17Detection JavaFX application.
 * <p>
 * This class is responsible for initializing global UI themes,
 * loading the primary FXML layout, and launching the main window.
 *
 * @author Le Ngoc Han (25han.ln@vinuni.edu.vn)
 * @version 1.0
 */
public class VF17Main extends Application {

    /**
     * JavaFX lifecycle method invoked automatically after application launch.
     * <p>
     * This method initializes MaterialFX and JavaFX themes, loads the main UI
     * from FXML, wires the controller lifecycle, and displays the primary stage.
     *
     * @param primaryStage the primary stage provided by the JavaFX runtime
     * @throws Exception if loading FXML or initializing the scene fails
     *
     * @author Le Ngoc Han (25han.ln@vinuni.edu.vn)
     * @version 1.0
     */
    @Override
    public void start(Stage primaryStage) throws Exception {
        primaryStage.setTitle("VF17Detection");

        UserAgentBuilder.builder()
                .themes(JavaFXThemes.MODENA)
                .themes(MaterialFXStylesheets.forAssemble(true))
                .setDeploy(true)
                .setResolveAssets(true)
                .build()
                .setGlobal();

        FXMLLoader loader = new FXMLLoader();
        loader.setLocation(getClass().getResource("/fxml/HomepageUI.fxml"));
        Parent root = loader.load();

        Scene scene = new Scene(root);
        primaryStage.setScene(scene);

        HomepageUIController controller = loader.getController();
        controller.handleShutdown(primaryStage);

        primaryStage.show();
    }

    /**
     * Standard Java entry point.
     * <p>
     * Delegates application startup to the JavaFX runtime, which
     * subsequently invokes {@link #start(Stage)}.
     *
     * @param args command-line arguments passed to the application
     *
     * @author Le Ngoc Han (25han.ln@vinuni.edu.vn)
     * @version 1.0
     */
    public static void main(String[] args) {
        launch(args);
    }
}
