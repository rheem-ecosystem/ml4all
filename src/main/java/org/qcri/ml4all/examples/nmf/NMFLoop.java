package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.Loop;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

public class NMFLoop extends Loop<Double, Tuple2<INDArray, INDArray>> {

    public int maxIterations;
    int currentIteration;
    double minRMSC;

    public NMFLoop(int maxIterations, double minRMSC) {
        this.maxIterations = maxIterations;
        this.minRMSC = minRMSC;
    }

    @Override
    public Double prepareConvergenceDataset(Tuple2 input, ML4allContext context) {
        return (double)context.getByKey("rmsc");
    }

    @Override
    public boolean terminate(Double input) {
        return (input < minRMSC || ++currentIteration >= maxIterations);
    }

}
