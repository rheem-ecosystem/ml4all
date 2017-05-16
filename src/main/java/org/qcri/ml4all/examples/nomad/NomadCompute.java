package org.qcri.ml4all.examples.nomad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.Compute;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

public class NomadCompute extends Compute<INDArray, INDArray> {


    @Override
    public INDArray process(INDArray input, ML4allContext context) {
        INDArray weights = (INDArray) context.getByKey("weights");

        return null;
    }
}
