package org.qcri.ml4all.examples.nomad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.Loop;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

import java.util.Random;

public class NomadLoop extends Loop<Tuple2<Tuple2<Integer, Integer>, Double>, INDArray> {

    public int maxIterations;
    int currentIteration;

    public NomadLoop(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    @Override
    public Tuple2 prepareConvergenceDataset(INDArray input, ML4allContext context) {
        int iteration = (int) context.getByKey("iter");
        INDArray a = (INDArray)context.getByKey("a");
        a.putColumn(0, input);
        context.put("a", a);
        Random rand = new Random();
        int i = rand.nextInt((int) context.getByKey("m"));
        int j = rand.nextInt((int) context.getByKey("n"));
        double datapoint = a.getDouble(i,j);
        //Tuple2uple2(new Tuple2(i,j), datapoint)

        return new Tuple2(new Tuple2(i,j), datapoint);
    }

    @Override
    public boolean terminate(Tuple2<Tuple2<Integer, Integer>, Double> input) {
        return ++currentIteration >= maxIterations;
    }

}
