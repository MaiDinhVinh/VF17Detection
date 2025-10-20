package com.arthroverse.vf17.microcontroller;

import com.fazecast.jSerialComm.SerialPort;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class ArduinoComm {
    private static final int  BAUD_RATE = 9600;
    private static final int NUM_DATA_BITS = 8;
    private static final int NUM_DATA_STOP_BITS = 1;
    private static final SerialPort PORT= SerialPort.getCommPort("/dev/tty.usbmodem1101");

    public static final void COMMUNICATE(boolean isRotten) throws IOException, InterruptedException {
        PORT.setBaudRate(BAUD_RATE);
        PORT.setNumDataBits(NUM_DATA_BITS);
        PORT.setNumStopBits(NUM_DATA_STOP_BITS);
        PORT.openPort();
        OutputStream outputStream = PORT.getOutputStream();
        byte[] data;
        if(isRotten){
            data = "1".getBytes(StandardCharsets.UTF_8);
        }else{
            data = "0".getBytes(StandardCharsets.UTF_8);
        }
        outputStream.write(data);
        outputStream.flush();
    }

//    public static final void RESET_SIGNAL() throws IOException{
//        doPause(20);
//        PORT.openPort();
//        OutputStream outputStream = PORT.getOutputStream();
//        byte[] data = "0".getBytes(StandardCharsets.UTF_8);
//        outputStream.write(data);
//        outputStream.flush();
//    }
//
//    private static void doPause(int ms) {
//        ScheduledExecutorService scheduledThreadPoolExecutor = Executors.newScheduledThreadPool(10);
//        try {
//            scheduledThreadPoolExecutor.schedule(() -> {
//            }, ms, TimeUnit.MILLISECONDS).get();
//        } catch (Exception e) {
//            throw new RuntimeException();
//        }
//    }
}
