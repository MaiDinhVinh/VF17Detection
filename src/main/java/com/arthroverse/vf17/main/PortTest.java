//this file is for testing only
//Author: Mai Dinh Vinh
package com.arthroverse.vf17.main;

import com.fazecast.jSerialComm.SerialPort;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

public class PortTest {
    private static final int  BAUD_RATE = 9600;
    private static final int NUM_DATA_BITS = 8;
    private static final int NUM_DATA_STOP_BITS = 1;
    private static final SerialPort PORT= SerialPort.getCommPort("/dev/tty.usbmodem21201");
    public static void main(String[] args) throws IOException {
        PORT.setBaudRate(BAUD_RATE);
        PORT.setNumDataBits(NUM_DATA_BITS);
        PORT.setNumStopBits(NUM_DATA_STOP_BITS);
        PORT.openPort();
        OutputStream outputStream = PORT.getOutputStream();
        byte[] data;
        data = "1".getBytes(StandardCharsets.UTF_8);
        outputStream.write(data);
        outputStream.flush();
    }
}
