package org.qcri.ml4all.examples.kmeans;

import org.qcri.ml4all.abstraction.api.Loop;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

import java.util.ArrayList;

/**
 * Created by zoi on 1/2/15.
 */
public class KmeansConvergeOrMaxIterationsLoop extends Loop<Double, ArrayList<Tuple2<Integer, double[]>>> {

    private double accuracy;
    private int maxIterations;

    private int currentIteration = 0;

    public KmeansConvergeOrMaxIterationsLoop(double accuracy, int maxIterations) {
        this.accuracy = accuracy;
        this.maxIterations = maxIterations;
    }

    @Override
    public Double prepareConvergenceDataset(ArrayList<Tuple2<Integer, double[]>> newCenters, ML4allContext context) {
        double[][] centers = (double[][]) context.getByKey("centers");
        double delta = 0.0;
        int dimension = centers[0].length;
        for (int i = 0; i < newCenters.size(); i++) {
            int centroidId = newCenters.get(i).field0;
            for (int j = 0; j < dimension; j++) {
                delta += Math.abs(centers[centroidId][j] - newCenters.get(i).field1[j]);
            }
        }
        return delta;
    }

    @Override
    public boolean terminate(Double input) {
        return input < accuracy || ++currentIteration >= maxIterations;
    }

}
