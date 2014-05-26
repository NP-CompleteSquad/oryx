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

package com.cloudera.oryx.rdf.serving.web;

import com.google.common.base.Preconditions;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Iterator;

import com.cloudera.oryx.rdf.common.tree.DecisionForest;
import com.cloudera.oryx.rdf.serving.generation.Generation;

/**
 * <p>Responds to a GET request to {@code /feature/importance} or {@code /feature/importance/[feature number]}.
 * In the first case, the results are feature importance values for all features in order as specified in
 * the configuration. In the second case, specified by number, the result is a single feature importance.</p>
 *
 * <p>In both cases, importance is given as a floating point value between 0 and 1 inclusive. Higher means more
 * important. The values are normalized and are not expressed in particular units.</p>
 *
 * @author Sean Owen
 */
public final class FeatureImportanceServlet extends AbstractRDFServlet {

  @Override
  protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
    CharSequence pathInfo = request.getPathInfo();
    String featureNumberString;
    if (pathInfo == null) {
      featureNumberString = null;
    } else {
      Iterator<String> pathComponents = SLASH.split(pathInfo).iterator();
      featureNumberString = pathComponents.hasNext() ? pathComponents.next() : null;
    }

    Generation generation = getGenerationManager().getCurrentGeneration();
    if (generation == null) {
      response.sendError(HttpServletResponse.SC_SERVICE_UNAVAILABLE,
                         "API method unavailable until model has been built and loaded");
      return;
    }

    DecisionForest forest = generation.getForest();
    double[] importances = forest.getFeatureImportances();

    if (featureNumberString == null) {

      output(request, response, importances);

    } else {

      int featureNumber;
      try {
        featureNumber = Integer.parseInt(featureNumberString);
        Preconditions.checkArgument(featureNumber >= 0);
        Preconditions.checkArgument(featureNumber < importances.length);
      } catch (IllegalArgumentException iae) {
        response.sendError(HttpServletResponse.SC_BAD_REQUEST, iae.toString());
        return;
      }

      output(request, response, new double[] { importances[featureNumber] });
    }
  }

}
