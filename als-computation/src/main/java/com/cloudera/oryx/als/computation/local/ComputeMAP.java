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

package com.cloudera.oryx.als.computation.local;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.Callable;

import com.cloudera.oryx.als.common.StringLongMapping;
import com.cloudera.oryx.common.collection.LongObjectMap;
import com.cloudera.oryx.common.collection.LongSet;
import com.cloudera.oryx.common.io.DelimitedDataUtils;
import com.cloudera.oryx.common.io.IOUtils;
import com.cloudera.oryx.common.iterator.FileLineIterable;
import com.cloudera.oryx.common.iterator.LongPrimitiveIterator;
import com.cloudera.oryx.common.math.SimpleVectorMath;

final class ComputeMAP implements Callable<Object> {

  private static final Logger log = LoggerFactory.getLogger(ComputeMAP.class);

  private final File testDir;
  private final LongObjectMap<float[]> X;
  private final LongObjectMap<float[]> Y;

  ComputeMAP(File testDir, LongObjectMap<float[]> X, LongObjectMap<float[]> Y) {
    this.testDir = testDir;
    this.X = X;
    this.Y = Y;
  }

  @Override
  public Object call() throws IOException {

    LongObjectMap<LongSet> testData = new LongObjectMap<LongSet>();

    for (File file : testDir.listFiles(IOUtils.NOT_HIDDEN)) {
      for (CharSequence line : new FileLineIterable(file)) {
        String[] columns = DelimitedDataUtils.decode(line);
        long userID = StringLongMapping.toLong(columns[0]);
        long itemID = StringLongMapping.toLong(columns[1]);
        LongSet itemIDs = testData.get(userID);
        if (itemIDs == null) {
          itemIDs = new LongSet();
          testData.put(userID, itemIDs);
        }
        itemIDs.add(itemID);
      }
    }

    Mean meanAveragePrecision = new Mean();

    LongPrimitiveIterator it = X.keySetIterator();
    while (it.hasNext()) {
      long userID = it.nextLong();
      float[] userVector = X.get(userID);

      LongSet ids = testData.get(userID);
      if (ids == null || ids.isEmpty()) {
        continue;
      }

      long[] itemIDs = ids.toArray();

      double[] scores = new double[itemIDs.length];
      for (int i = 0; i < itemIDs.length; i++) {
        long itemID = itemIDs[i];
        float[] itemVector = Y.get(itemID);
        if (itemVector == null) {
          continue;
        }
        scores[i] = SimpleVectorMath.dot(userVector, itemVector);
      }

      int[] rank = new int[itemIDs.length];

      for (LongObjectMap.MapEntry<float[]> entry : Y.entrySet()) {
        double score = SimpleVectorMath.dot(userVector, entry.getValue());
        for (int i = 0; i < itemIDs.length; i++) {
          if (score > scores[i]) {
            rank[i]++;
          }
        }
      }

      Arrays.sort(rank);

      Mean precision = new Mean();
      double totalPrecisionTimesRelevance = 0.0;
      for (int i = 0; i < rank.length; i++) {
        int relevantRetrieved = i + 1;
        int precisionAt = rank[i] + 1;
        precision.increment((double) relevantRetrieved / precisionAt);
        totalPrecisionTimesRelevance += precision.getResult();
      }
      double averagePrecision = totalPrecisionTimesRelevance / rank.length;

      meanAveragePrecision.increment(averagePrecision);
    }

    log.info("Mean average precision: {}", meanAveragePrecision.getResult());

    return null;
  }

}
