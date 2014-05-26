/*
 * Copyright (c) 2014, Cloudera, Inc. All Rights Reserved.
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

package com.cloudera.oryx.als.computation.merge;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.crunch.Emitter;
import org.apache.crunch.Pair;

import com.cloudera.oryx.common.random.RandomManager;
import com.cloudera.oryx.computation.common.fn.OryxDoFn;

public final class SplitTestFn extends OryxDoFn<String, Pair<String, String>> {

  private final double testSetFraction;
  private RandomGenerator random;

  public SplitTestFn(double testSetFraction) {
    this.testSetFraction = testSetFraction;
  }

  @Override
  public void initialize() {
    super.initialize();
    random = RandomManager.getRandom();
  }

  @Override
  public void process(String line, Emitter<Pair<String, String>> emitter) {
    Pair<String, String> out;
    if (testSetFraction > 0.0 && random.nextDouble() < testSetFraction) {
      out = Pair.of(null, line);
    } else {
      out = Pair.of(line, null);
    }
    emitter.emit(out);
  }

}
