package org.qcri.ml4all.examples.sgd;

import org.qcri.ml4all.abstraction.api.LocalStage;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

/**
 * Created by zoi on 22/1/15.
 */
public class SGDStageWithZeros extends LocalStage {

    int dimension;

    public SGDStageWithZeros(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public void staging (ML4allContext context) {
        double[] weights = new double[dimension];
        context.put("weights", weights);
        context.put("iter", 1);
    }
}
