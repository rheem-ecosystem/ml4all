package org.qcri.ml4all.examples.nomad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.Loop;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

public class NomadLoop extends Loop<Double, INDArray> {

    public int maxIterations;
    int currentIteration;

    public NomadLoop(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    @Override
    public Double prepareConvergenceDataset(INDArray input, ML4allContext context) {
        return null;
    }

    @Override
    public boolean terminate(Double input) {
        return (++currentIteration >= maxIterations);
    }

}
