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

package com.cloudera.oryx.kmeans.computation.local;

import com.cloudera.oryx.common.math.NamedRealVector;
import com.cloudera.oryx.common.settings.ConfigUtils;
import com.cloudera.oryx.common.settings.InboundSettings;
import com.cloudera.oryx.kmeans.common.Distance;
import com.cloudera.oryx.kmeans.computation.evaluate.KMeansEvaluationData;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.typesafe.config.Config;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.Callable;

public class Assignment implements Callable<List<String>> {

  private static final Logger log = LoggerFactory.getLogger(Assignment.class);
  private static final Joiner COMMA = Joiner.on(',');

  private final List<List<RealVector>> folds;
  private final List<KMeansEvaluationData> clusters;

  public Assignment(List<List<RealVector>> folds, List<KMeansEvaluationData> clusters) {
    this.folds = folds;
    this.clusters = clusters;
  }

  @Override
  public List<String> call() {
    Config config = ConfigUtils.getDefaultConfig();
    List<String> ret = Lists.newArrayList();
    if (doOutlierComputation(config)) {
      InboundSettings inboundSettings = InboundSettings.create(config);
      if (inboundSettings.getIdColumns().isEmpty()) {
        log.error("Cluster assignments require that id-columns be configured for each vector");
        return ImmutableList.of();
      }
      for (List<RealVector> fold : folds) {
        for (RealVector vec : fold) {
          NamedRealVector nvec = (NamedRealVector) vec;
          for (KMeansEvaluationData data : clusters) {
            Distance d = data.getBest().getDistance(nvec);
            ret.add(COMMA.join(
                nvec.getName(), data.getK(), d.getClosestCenterId(), d.getSquaredDistance()));
          }
        }
      }
      return ret;
    }
    return ImmutableList.of();
  }

  private static boolean doOutlierComputation(Config config) {
    return config.hasPath("model.outliers") && config.getBoolean("model.outliers.compute");
  }
}
