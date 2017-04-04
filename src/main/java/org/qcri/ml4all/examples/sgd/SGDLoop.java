package org.qcri.ml4all.examples.sgd;

import org.qcri.ml4all.abstraction.api.Loop;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

public class SGDLoop extends Loop<Double, double[]> {

    public double accuracy;
    public int maxIterations;
    int currentIteration;

    public SGDLoop(double accuracy, int maxIterations) {
        this.accuracy = accuracy;
        this.maxIterations = maxIterations;
    }

    @Override
    public Double prepareConvergenceDataset(double[] input, ML4allContext context) {
        double[] weights = (double[]) context.getByKey("weights");
        double delta = 0.0;
        for (int j = 0; j < weights.length; j++) {
            delta += Math.abs(weights[j] - input[j]);
        }
        return delta;
    }

    @Override
    public boolean terminate(Double input) {
        return (input < accuracy || ++currentIteration >= maxIterations);
    }

}
