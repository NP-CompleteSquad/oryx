/**
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
package com.cloudera.oryx.kmeans.common;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Chooses the best clustering based on the lowest-cost solution found for the most stable value of K, where
 * the stability of K is determined by taking the mean or the median of one of the main cluster stability metric
 * (either Van Dongen or the Variation of Information.)
 */
public final class MostStableEvalStrategy implements KMeansEvalStrategy {

  private static final Logger log = LoggerFactory.getLogger(MostStableEvalStrategy.class);

  private final boolean useMedian;
  private final boolean useVariationOfInformation;

  public MostStableEvalStrategy(boolean useMedian, boolean useVariationOfInformation) {
    this.useMedian = useMedian;
    this.useVariationOfInformation = useVariationOfInformation;
  }

  @Override
  public List<ClusterValidityStatistics> evaluate(List<ClusterValidityStatistics> stats) {
    Map<Integer, ClusterValidityStatistics> bestPerK = Maps.newHashMap();
    Map<Integer, List<Double>> metrics = Maps.newHashMap();
    for (ClusterValidityStatistics stat : stats) {
      if (stat.getK() > 1) {
        ClusterValidityStatistics best = bestPerK.get(stat.getK());
        if (best == null || stat.getTotalCost() < best.getTotalCost()) {
          bestPerK.put(stat.getK(), stat);
        }
        List<Double> m = metrics.get(stat.getK());
        if (m == null) {
          m = Lists.newArrayList();
          metrics.put(stat.getK(), m);
        }
        m.add(useVariationOfInformation ? stat.getVariationOfInformation() : stat.getVanDongen());
      }
    }
    int bestK = getBestK(metrics);
    log.info("Best K = {}", bestK);
    ClusterValidityStatistics cvs = bestPerK.get(bestK);
    if (cvs == null) {
      log.warn("No ClusterValidityStatistics found for best k, returning");
      return ImmutableList.of();
    } else {
      return ImmutableList.of(cvs);
    }
  }

  private int getBestK(Map<Integer, List<Double>> metrics) {
    int bestK = -1;
    double bestScore = Double.POSITIVE_INFINITY;
    for (Map.Entry<Integer, List<Double>> e : metrics.entrySet()) {
      double score = useMedian ? getMedian(e.getValue()) : getMean(e.getValue());
      if (score < bestScore) {
        bestScore = score;
        bestK = e.getKey();
      }
    }
    return bestK;
  }

  private static double getMean(Collection<Double> values) {
    double sum = 0.0;
    for (double d : values) {
      sum += d;
    }
    return sum / values.size();
  }

  private static double getMedian(List<Double> values) {
    int n = values.size();
    Collections.sort(values);
    if (n % 2 == 0) {
      return (values.get((n / 2) - 1) + values.get(n / 2)) / 2.0;
    } else {
      return values.get(n / 2);
    }
  }
}
