package org.qcri.ml4all.examples.nomad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.qcri.ml4all.abstraction.api.LocalStage;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

/**
 * Created by zoi on 22/1/15.
 */
public class NomadStageWithRandomValues extends LocalStage {

    int dimension;
    int nRows;
    int nColumns;

    public NomadStageWithRandomValues(int dimension, int nColumns, int nRows) {
        this.dimension = dimension;
        this.nRows = nRows;
        this.nColumns = nColumns;
    }

    @Override
    public void staging (ML4allContext context) {
        //double[] weights = new double[dimension];
      //  INDArray allZeros = Nd4j.zeros(nRows, nColumns);

        int[] shape = new int[]{nRows, nColumns};
        INDArray weights = Nd4j.rand(shape);

        context.put("weights", weights);
        context.put("iter", 1);
    }
}
