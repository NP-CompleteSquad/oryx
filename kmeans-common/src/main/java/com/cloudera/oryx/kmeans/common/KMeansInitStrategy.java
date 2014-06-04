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

package com.cloudera.oryx.kmeans.common;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * The strategies used to choose the initial {@link Centers} used for a k-means algorithm
 * run prior to running Lloyd's algorithm.
 */
public enum KMeansInitStrategy {

  /**
   * Uses the classic random selection strategy to create the initial {@code Centers}. The
   * algorithm will randomly choose K points from the input points, favoring points with
   * higher weights.
   */
  RANDOM {
    @Override
    public <W extends Weighted<RealVector>> Centers apply(List<W> points,
                                                          int numClusters,
                                                          RandomGenerator random) {
      return new Centers(
          Lists.transform(Weighted.sample(points, numClusters, random), new Function<W, RealVector>() {
            @Override
            public RealVector apply(W wt) {
              return wt.thing();
            }
          }));
    }
  },
  
  /**
   * Uses the <i>k-means++</i> strategy described in Arthur and Vassilvitskii (2007).
   * See <a href="http://en.wikipedia.org/wiki/K-means%2B%2B">the Wikipedia page</a>
   * for details.
   */
  PLUS_PLUS {
    @Override
    public <W extends Weighted<RealVector>> Centers apply(List<W> points,
                                                int numClusters,
                                                RandomGenerator random) {
      Centers centers = RANDOM.apply(points, 1, random);
      double[] cumulativeScores = new double[points.size() + 1];
      for (int i = 1; i < numClusters; i++) {
        cumulativeScores[0] = 0;
        for (int j = 0; j < points.size(); j++) {
          W wv = points.get(j);
          double score = centers.getDistance(wv.thing()).getSquaredDistance() * wv.weight();
          cumulativeScores[j + 1] = cumulativeScores[j] + score;
        }
        double r = cumulativeScores[points.size()] * random.nextDouble();
        int next = Arrays.binarySearch(cumulativeScores, r);
        int index = (next > 0) ? next - 1 : -2 - next;
        while (index > 0 && centers.contains(points.get(index).thing())) {
          index--;
        }
        centers = centers.extendWith(points.get(index).thing());
      }
      return centers;
    }
  },
    /**
     * Uses the <i>k-means||</i> strategy proposed by Bahman Bahmani from Standford University.
     * See <a href="http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf">the Related Paper</a>
     * for details.
     */
    PARALLEL {
                @Override
                public <W extends Weighted<RealVector>> Centers apply(List<W> points,
                                                                      int numClusters,
                                                                      RandomGenerator random) {

            /*
             * Algorithm  k-means||(k,l) initialization.
             *  1: C   sample a point uniformly at random from X
             *  2:  ψ<-- ΦX(C)
             *  3: for O(logψ ) times do
             *  4: C'<--sample each point x ε X independently with
             *     probability px = ld^2(x0,C)/ΦX(C)
             *  5: C<--C U C'
             *  6: end for
             *  7: For x ε C, set wx to be the number of points in X closer
             *     to x than any other point in C
             *  8: Recluster the weighted points in C into k clusters
             */
                    Centers intermediateCenters = RANDOM.apply(points, 1, random);
                    Random rand = new Random();
                    double[] cumulativeScores = new double[points.size() + 1];
                    long oversamplingFactor = Math.round(0.5*numClusters);

                    //Calculate the initial cost of clustering after selecting the 1st center.
                    double costOfClustering=intermediateCenters.getClusteringCost(points);

                    //Oversampling Phase.
                    for(int i=0; i< Math.round(Math.log10(costOfClustering)); i++) {
                        cumulativeScores[0] = 0;
                        for (int j = 0; j < points.size(); j++) {
                            W weightedVec = points.get(j);
                            double score = oversamplingFactor * intermediateCenters.getDistance(weightedVec.thing()).getSquaredDistance() * weightedVec.weight();
                            cumulativeScores[j + 1] = cumulativeScores[j] + score;
                        }
                        for (int l = 0; l < oversamplingFactor; l++) {
                            double r = cumulativeScores[points.size()] * random.nextDouble();
                            int next = Arrays.binarySearch(cumulativeScores, r);
                            int index = (next > 0) ? next - 1 : -2 - next;
                            while (index > 0 && intermediateCenters.contains(points.get(index).thing())) {
                                index--;
                            }
                            intermediateCenters = intermediateCenters.extendWith(points.get(index).thing());
                        }
                    }
                    //Count points closer to a specific center than any other center in the intermediate set.
                    List<WeightedRealVector> centersToRecluster=Lists.newArrayList();
                    double[] centerIdFrequency = new double[intermediateCenters.size()];
                    for (W weightedVec : points) {
                        centerIdFrequency[intermediateCenters.getDistance(weightedVec.thing()).getClosestCenterId()]++;
                    }
                    for(int i=0; i<intermediateCenters.size();i++){
                        RealVector center=intermediateCenters.get(i);
                        centersToRecluster.add(new WeightedRealVector(center,centerIdFrequency[i]));
                    }
                    //Recluster-Reduce Phase.
                    return PLUS_PLUS.apply(centersToRecluster,numClusters,random);
                }
            };

  /**
   * Use this instance to create the initial {@code Centers} from the given parameters.
   * 
   * @param points The candidate {@code WeightedVec} instances for the cluster
   * @param numClusters The number of points in the center (i.e., the "k" in "k-means")
   * @param randomGenerator The {@code RandomGenerator} instance to use
   * @return A new {@code Centers} instance created using this instance
   */
  public abstract <W extends Weighted<RealVector>> Centers apply(List<W> points, int numClusters,
                                                       RandomGenerator randomGenerator);
}
