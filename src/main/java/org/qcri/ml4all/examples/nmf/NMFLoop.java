package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.Loop;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

public class NMFLoop extends Loop<Tuple2<INDArray, INDArray>, Tuple2<INDArray, INDArray>> {

    public int maxIterations;
    int currentIteration;

    public NMFLoop(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    @Override
    public Tuple2 prepareConvergenceDataset(Tuple2 input, ML4allContext context) {
        return new Tuple2(input.getField0(), input.getField1());
    }

    @Override
    public boolean terminate(Tuple2 input) {
        return ++currentIteration >= maxIterations;
    }

}
