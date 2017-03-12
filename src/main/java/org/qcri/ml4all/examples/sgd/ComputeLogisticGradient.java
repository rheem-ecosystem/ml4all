package org.qcri.ml4all.examples.sgd;

import org.qcri.ml4all.abstraction.api.Compute;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

import java.util.Arrays;

public class ComputeLogisticGradient extends Compute<double[], double[]> {


    @Override
    public double[] process(double[] point, ML4allContext context) {

        double[] weights = (double[]) context.getByKey("weights");
        double[] gradient = new double[point.length];
        double dot = 0;
        for (int j = 0; j < weights.length; j++)
            dot += weights[j] * point[j + 1];

        for (int j = 0; j < weights.length; j++)
            gradient[j + 1] = ((1 / (1 + Math.exp(-1 * dot))) - point[0]) * point[j + 1];

        gradient[0] = 1; //counter for the step size required in the update

        return gradient;
    }

}
