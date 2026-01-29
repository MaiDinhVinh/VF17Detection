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
 * File Name:       [PortTest.java]
 * Developer:       Dinh Hieu Minh
 * Description:     [Standalone utility class to test Arduino serial communication
 *                  and reset the servo motor to its default (0-degree) position]
 ******************************************************************************/

package com.arthroverse.vf17.main;

import com.fazecast.jSerialComm.SerialPort;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Provides a simple executable entry point to verify Arduino serial communication
 * using jSerialComm.
 * <p>
 * This class is intended for hardware testing purposes only. When executed,
 * it sends a reset signal to the Arduino controller to return the servo motor
 * to its default position (0 degrees).
 *
 * @author Dinh Hieu Minh (25minh.dh2@vinuni.edu.vn)
 * @version 1.0
 */
public class PortTest {

    /**
     * Baud rate used for serial communication with the Arduino controller.
     */
    private static final int BAUD_RATE = 9600;

    /**
     * Number of data bits used in serial communication.
     */
    private static final int NUM_DATA_BITS = 8;

    /**
     * Number of stop bits used in serial communication.
     */
    private static final int NUM_DATA_STOP_BITS = 1;

    /**
     * Serial port instance representing the connected Arduino device.
     */
    private static final SerialPort PORT =
            SerialPort.getCommPort("/dev/tty.usbmodem11101");

    /**
     * Application entry point.
     * <p>
     * This method initializes the serial port configuration, opens the connection,
     * and sends a reset command ({@code "1"}) to the Arduino controller. The Arduino
     * firmware interprets this signal as an instruction to reset the servo motor
     * back to the 0-degree position.
     *
     * @param args command-line arguments (not used)
     * @throws IOException if an I/O error occurs while writing to the serial port
     *
     * @author Dinh Hieu Minh (25minh.dh2@vinuni.edu.vn)
     * @version 1.0
     */
    public static void main(String[] args) throws IOException {
        PORT.setBaudRate(BAUD_RATE);
        PORT.setNumDataBits(NUM_DATA_BITS);
        PORT.setNumStopBits(NUM_DATA_STOP_BITS);

        PORT.openPort();

        OutputStream outputStream = PORT.getOutputStream();

        byte[] data = "1".getBytes(StandardCharsets.UTF_8);
        outputStream.write(data);
        outputStream.flush();
    }
}
