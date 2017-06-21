package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.Loop;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

import java.util.ArrayList;
import java.util.List;

public class NMFLoop extends Loop<Double, Double> {

    int maxIterations;
    int currentIteration;
    double minRMSC;
    double currentRMSC;
    int epoch;
    List<Double> items;
    int epoch_Size ;
    int currentItr = 1;
    public NMFLoop(int datasetSize, int features, double minRMSC, int epoch) {
        this.maxIterations = epoch * (datasetSize*features);
        this.minRMSC = minRMSC;
        this.epoch = epoch;
        this.items = new ArrayList<>();
        this.epoch_Size = datasetSize*features;
    }

    @Override
    public Double prepareConvergenceDataset(Double input, ML4allContext context) {
        // calculate per epoch
        double rmsc = 0.0;

        int k = this.currentItr % epoch_Size;
        if(k == 0) {
            items.add(input);
            if(items.size() > 0){
                double s =0.0;
                for(double item : items){
                    s = s + Math.pow(item, 2);
                }

                double s2 = s/items.size();
                items.clear();
                rmsc =  Math.sqrt(s2);
                items.clear();
            }
        }
        this.currentItr++;
        System.out.println("prepareConvergenceDataset:" + maxIterations + " , " + rmsc);
        return rmsc;

    }

    @Override
    public boolean terminate(Double input) {
      //  return (input >= minRMSC || ++currentIteration >= maxIterations);
        System.out.println(maxIterations + " , " + currentIteration);
        return (++currentIteration >= maxIterations);
    }


}
