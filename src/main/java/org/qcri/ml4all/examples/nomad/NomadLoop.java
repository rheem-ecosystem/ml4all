package org.qcri.ml4all.examples.nomad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.Loop;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

import java.util.Random;

public class NomadLoop extends Loop<Tuple2<INDArray, INDArray>, Tuple2<INDArray, INDArray>> {

    public int maxIterations;
    int currentIteration;

    public NomadLoop(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    @Override
    public Tuple2 prepareConvergenceDataset(Tuple2 input, ML4allContext context) {
        int iteration = (int) context.getByKey("iter");
        return new Tuple2(input.getField0(), input.getField1());
    }

    @Override
    public boolean terminate(Tuple2 input) {
        return ++currentIteration >= maxIterations;
    }

}
