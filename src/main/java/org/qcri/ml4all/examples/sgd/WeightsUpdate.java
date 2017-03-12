package org.qcri.ml4all.examples.sgd;

import org.qcri.ml4all.abstraction.api.UpdateLocal;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

public class WeightsUpdate extends UpdateLocal<double[], double[]> {

    double[] weights;
    int current_iteration;

    double stepSize = 1;
    double regulizer = 0;

    public WeightsUpdate () { }

    public WeightsUpdate (double stepSize, double regulizer) {
        this.stepSize = stepSize;
        this.regulizer = regulizer;
    }

    @Override
    public double[] process(double[] input, ML4allContext context) {
        double[] weights = (double[]) context.getByKey("weights");
        double count = input[0];
        int current_iteration = (int) context.getByKey("iter");
        double alpha = stepSize / current_iteration;
        double[] newWeights = new double[weights.length];
        for (int j = 0; j < weights.length; j++) {
            newWeights[j] = (1 - alpha * regulizer) * weights[j] - alpha * (1.0 / count) * input[j + 1];
        }
        return newWeights;
    }

    @Override
    public ML4allContext assign(double[] input, ML4allContext context) {
        context.put("weights", input);
        int iteration = (int) context.getByKey("iter");
        context.put("iter", ++iteration);
        return context;
    }


}
