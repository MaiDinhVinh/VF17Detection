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
 * File Name:       [Reusable.java]
 * Developer:       Mai Dinh Vinh
 * Description:     [All reusable elements should be declared here to improve the
 *                  modularity of the program]
 ******************************************************************************/
package com.arthroverse.vf17.utilities;

import javafx.scene.control.TextArea;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Enumeration that defines reusable UI components shared across the application.
 * <p>
 * This enum follows the singleton-like enum pattern to ensure that each reusable
 * component is instantiated exactly once and safely shared across different
 * parts of the system.
 * <p>
 * Currently, it provides a reusable {@link TextArea} container used for displaying
 * detailed exception stack traces in alert dialogs.
 *
 * @author Mai Dinh Vinh (25vinh.md@vinuni.edu.vn)
 * @version 1.0
 */
public enum Reusable {

    /**
     * Reusable container used to display exception stack traces
     * inside JavaFX alert dialogs.
     */
    ALERT_CONTAINER();

    /**
     * Text area that holds stack trace content for error alerts.
     * <p>
     * This component is configured to be non-editable, wrap text automatically,
     * and have a predefined preferred size suitable for displaying long stack traces.
     */
    private TextArea stackTraceAlertContainer;

    /**
     * Private constructor that initializes reusable UI components.
     * <p>
     * The constructor is invoked exactly once per enum constant,
     * guaranteeing a single shared instance of each reusable element.
     *
     * @author Mai Dinh Vinh (25vinh.md@vinuni.edu.vn)
     * @version 1.0
     */
    private Reusable(){
        stackTraceAlertContainer = new TextArea();
        stackTraceAlertContainer.setEditable(false);
        stackTraceAlertContainer.setWrapText(true);
        stackTraceAlertContainer.setPrefWidth(850);
        stackTraceAlertContainer.setPrefHeight(400);
    }

    /**
     * Returns the reusable {@link TextArea} instance associated with this enum constant.
     * <p>
     * This method is primarily used by alert utilities to embed the text area
     * into JavaFX dialog panes for displaying exception details.
     *
     * @return the reusable {@link TextArea} instance
     *
     * @author Mai Dinh Vinh (25vinh.md@vinuni.edu.vn)
     * @version 1.0
     */
    public TextArea getObject(){
        return this.stackTraceAlertContainer;
    }
}
