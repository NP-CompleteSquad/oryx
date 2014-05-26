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

package com.cloudera.oryx.common.parallel;

import org.junit.Test;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import com.cloudera.oryx.common.OryxTest;

/**
 * Tests {@link ExecutorUtils}.
 *
 * @author Sean Owen
 */
public final class ExecutorUtilsTest extends OryxTest {

  @Test
  public void testNow() {
    final AtomicInteger count = new AtomicInteger();
    ExecutorService executor = Executors.newSingleThreadExecutor();
    executor.submit(new Callable<Object>() {
      @Override
      public Void call() throws InterruptedException {
        Thread.sleep(500L);
        count.incrementAndGet();
        return null;
      }
    });
    ExecutorUtils.shutdownNowAndAwait(executor);
    assertEquals(0, count.get());
  }

  @Test
  public void testGetResults() {
    ExecutorService executor = Executors.newSingleThreadExecutor();
    Future<Integer> future = executor.submit(new Callable<Integer>() {
      @Override
      public Integer call() {
        return 1;
      }
    });
    List<Integer> result = ExecutorUtils.getResults(Collections.singletonList(future));
    ExecutorUtils.shutdownNowAndAwait(executor);
    assertEquals(1, result.size());
    assertEquals(1, result.get(0).intValue());
  }

  @Test(expected = IllegalStateException.class)
  public void testCheckExceptions() {
    ExecutorService executor = Executors.newSingleThreadExecutor();
    try {
      Future<Object> future = executor.submit(new Callable<Object>() {
        @Override
        public Void call() throws IOException {
          throw new IOException();
        }
      });
      ExecutorUtils.getResults(Collections.singletonList(future));
    } finally {
      ExecutorUtils.shutdownNowAndAwait(executor);
    }
  }

}
