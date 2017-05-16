package org.qcri.ml4all.examples.nomad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.UpdateLocal;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

public class NomadUpdate extends UpdateLocal<INDArray, INDArray> {

    double stepSize = 1;
    double regulizer = 0;

    public NomadUpdate() { }

    public NomadUpdate(double stepSize, double regulizer) {
        this.stepSize = stepSize;
        this.regulizer = regulizer;
    }


    @Override
    public INDArray process(INDArray input, ML4allContext context) {
        return null;
    }

    @Override
    public ML4allContext assign(INDArray input, ML4allContext context) {
        return null;
    }
}
