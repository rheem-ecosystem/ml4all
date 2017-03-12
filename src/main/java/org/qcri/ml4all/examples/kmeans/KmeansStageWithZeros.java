package org.qcri.ml4all.examples.kmeans;

import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.ml4all.abstraction.api.LocalStage;

/**
 * Created by zoi on 22/1/15.
 */
public class KmeansStageWithZeros extends LocalStage {

    int k, dimension;

    public KmeansStageWithZeros(int k, int dimension) {
        this.k = k;
        this.dimension = dimension;
    }

    @Override
    public void staging (ML4allContext context) {
        double[][] centers = new double[k][];
        for (int i = 0; i < k; i++) {
            centers[i] = new double[dimension];
        }
        context.put("centers", centers);
    }
}
