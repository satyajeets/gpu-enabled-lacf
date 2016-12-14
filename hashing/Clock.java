/*
 * Clock.java    1.0 2000/01/01
 *
 * Copyright (c) 2000 Stefan Nilsson
 * KTH, Nada
 * SE-100 44 Stockholm, Sweden
 * http://www.nada.kth.se/~snilsson
 *
 * The code presented in this file has been tested with care but
 * is not guaranteed for any purpose. The writer does not offer
 * any warranties nor does he accept any liabilities with respect
 * to the code.
 */


import java.io.Serializable;

/**
 * A simple clock utility for measuring time in milliseconds.
 *
 * @author  Stefan Nilsson
 * @version 1.0, 1 jan 2000
 */
public class Clock implements Serializable {
    /**
     * Time when start() was called. Contains a valid time
     * only if the clock is running.
     */
    private long startTime;

    /**
     * Holds the total accumulated time since last reset.
     * Does not include time since start() if clock is running.
     */
    private long totalTime = 0;

    private boolean isRunning = false;

    /**
     * Turns this clock on.
     * Has no effect if the clock is already running.
     */
    public void start() {
        if (!isRunning) {
            isRunning = true;
            startTime = System.currentTimeMillis();
        }
    }

    /**
     * Turns this clock off.
     * Has no effect if the clock is not running.
     */
    public void stop() {
        if (isRunning) {
            totalTime += System.currentTimeMillis() - startTime;
            isRunning = false;
        }
    }

    /**
     * Resets this clock.
     * The clock is stopped and the total time is set to 0.
     */
    public void reset() {
        isRunning = false;
        totalTime = 0;
    }

    /**
     * Returns the total time that this clock has been running since
     * last reset.
     * Does not affect the running status of the clock; if the clock
     * is running when this method is called, it continues to run.
     *
     * @return the time in milliseconds.
     */
    public long getTime() {
        return totalTime +
            (isRunning ? System.currentTimeMillis() - startTime : 0);
    }

    /**
     * Tests if this clock is running.
     *
     * @return <tt>true</tt> if this clock is running;
     *         <tt>false</tt> otherwise.
     */
    public boolean isRunning() {
        return isRunning;
    }
}