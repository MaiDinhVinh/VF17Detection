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
 * Date Created:    [29-1-2026]
 * Last Modified:   [29-1-2026]
 * <p>
 * File Name:       [AlertUtil.java]
 * Developer:       Mai Dinh Vinh
 * Description:     [Interface to display all kinds of alerts to user]
 ******************************************************************************/
package com.arthroverse.vf17.utilities;

import javafx.application.Platform;
import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Optional;

/**
 * Utility interface that provides reusable helper methods for displaying
 * JavaFX alert dialogs related to exception handling and error reporting.
 * <p>
 * This utility ensures that all UI-related alert operations are executed
 * safely on the JavaFX Application Thread by using {@link Platform#runLater(Runnable)}.
 * <p>
 * The interface is designed to be stateless and provides only static methods.
 *
 * @author Mai Dinh Vinh (25vinh.md@vinuni.edu.vn)
 * @version 1.0
 */
public abstract interface AlertUtil {

    /**
     * Displays an error alert dialog that shows a detailed exception stack trace
     * to the user in a reusable UI container.
     * <p>
     * This method is thread-safe and can be called from any background thread.
     * Internally, it delegates the UI rendering logic to the JavaFX Application
     * Thread using {@link Platform#runLater(Runnable)}.
     *
     * @param stackTraceAsString the stack trace of the exception formatted as a {@link String}
     * @param title the header text displayed at the top of the alert dialog
     *
     * @author Mai Dinh Vinh (25vinh.md@vinuni.edu.vn)
     * @version 1.0
     */
    public static void generateExceptionViewer(String stackTraceAsString, String title){
        Platform.runLater(() -> {
            Alert alert = new Alert(Alert.AlertType.ERROR);
            alert.setTitle("VF17Detection");
            alert.setHeaderText(title);
            Reusable.ALERT_CONTAINER.getObject().setText(stackTraceAsString);
            alert.getDialogPane().setContent(Reusable.ALERT_CONTAINER.getObject());
            alert.showAndWait();
        });
    }

    /**
     * Converts a {@link Throwable} into a formatted stack trace string.
     * <p>
     * This method is typically used to capture detailed exception information
     * for logging or for displaying in an error dialog via
     * {@link #generateExceptionViewer(String, String)}.
     *
     * @param t the {@link Throwable} whose stack trace is to be converted
     * @return a {@link String} containing the full stack trace of the exception
     *
     * @author Mai Dinh Vinh (25vinh.md@vinuni.edu.vn)
     * @version 1.0
     */
    public static String generateExceptionString(Throwable t){
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);
        t.printStackTrace(pw);
        return sw.toString();
    }
}
