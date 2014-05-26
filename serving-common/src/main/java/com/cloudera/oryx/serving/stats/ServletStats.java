/*
 * Copyright (c) 2013, Cloudera, Inc. All Rights Reserved.
 *
 * Cloudera, Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"). You may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for
 * the specific language governing permissions and limitations under the
 * License.
 */

package com.cloudera.oryx.serving.stats;

import java.io.Serializable;
import java.util.concurrent.TimeUnit;

import com.cloudera.oryx.common.stats.RunningStatistics;
import com.cloudera.oryx.common.stats.RunningStatisticsPerTime;

/**
 * @author Sean Owen
 */
public final class ServletStats implements Serializable {

  private final RunningStatistics allTimeNanosec;
  private final RunningStatisticsPerTime lastHourNanosec;
  private int numClientErrors;
  private int numServerErrors;

  public ServletStats() {
    allTimeNanosec = new RunningStatistics();
    lastHourNanosec = new RunningStatisticsPerTime(TimeUnit.HOURS);
  }

  public synchronized void addTimingNanosec(long timingNanosec) {
    allTimeNanosec.increment(timingNanosec);
    lastHourNanosec.increment(timingNanosec);
  }

  public synchronized RunningStatistics getAllTimeNanosec() {
    return allTimeNanosec;
  }

  public synchronized RunningStatisticsPerTime getLastHourNanosec() {
    return lastHourNanosec;
  }

  public synchronized int getNumClientErrors() {
    return numClientErrors;
  }

  public synchronized void incrementClientErrors() {
    numClientErrors++;
  }

  public synchronized int getNumServerErrors() {
    return numServerErrors;
  }

  public synchronized void incrementServerErrors() {
    numServerErrors++;
  }

}
