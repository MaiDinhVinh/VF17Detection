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
 * File Name:       [ArduinoComm.java]
 * Developer:       Dinh Hieu Minh
 * Description:     [Java communication with Arduino Controller using jSerialComm]
 ******************************************************************************/

package com.arthroverse.vf17.microcontroller;

import com.arthroverse.vf17.utilities.AlertUtil;
import com.fazecast.jSerialComm.SerialPort;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Provides a minimal utility to send serial commands from Java to an Arduino controller
 * using the jSerialComm library.
 *
 * @author Dinh Hieu Minh (25minh.dh2@vinuni.edu.vn)
 * @version 1.0
 */
public class ArduinoComm {

    private static final int BAUD_RATE = 9600;
    private static final int NUM_DATA_BITS = 8;
    private static final int NUM_DATA_STOP_BITS = 1;
    private static SerialPort PORT = null;

    /**
     * Sends a single-byte command to Arduino via serial port to indicate whether the current
     * detected object is rotten.
     * <p>
     * This method writes {@code "1"} if {@code isRotten} is {@code true}, otherwise writes {@code "0"}.
     * <p>
     * If any exception due to Arduino connection are thrown, the catch clause will transform the error
     * message to String and then displayed through JavaFX element for user to alert them about the
     * incident
     *
     * @param isRotten {@code true} to send rotten signal; {@code false} to send fresh signal
     *
     * @author Dinh Hieu Minh (25minh.dh2@vinuni.edu.vn)
     * @version 2.0
     */
    public static final void COMMUNICATE(boolean isRotten) {
        try{
            PORT = SerialPort.getCommPort("/dev/tty.usbmodem11101");
            PORT.setBaudRate(BAUD_RATE);
            PORT.setNumDataBits(NUM_DATA_BITS);
            PORT.setNumStopBits(NUM_DATA_STOP_BITS);
            PORT.openPort();

            OutputStream outputStream = PORT.getOutputStream();
            byte[] data;

            if (isRotten) {
                data = "1".getBytes(StandardCharsets.UTF_8);
            } else {
                data = "0".getBytes(StandardCharsets.UTF_8);
            }

            outputStream.write(data);
            outputStream.flush();
        }catch(IOException e){
            AlertUtil.generateExceptionViewer(AlertUtil.generateExceptionString(e),
                    "Fatal error on actuator circuit, please check the circuit and restart the system!");
        }
    }
}
