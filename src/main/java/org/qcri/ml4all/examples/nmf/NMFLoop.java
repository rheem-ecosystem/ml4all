package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.Loop;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class NMFLoop extends Loop<Double, Double> {


    long currentIteration = 1;
    long maxIternation;

    int epoch;
    List<Double> items;
    long perEpochSize ;
    int alphaEpoch;

    public NMFLoop(int datasetSize, int featureSize,int epoch, int alphaEpoch) {
        this.perEpochSize = datasetSize*featureSize;
        this.maxIternation = epoch * (datasetSize * featureSize);
        this.items = new ArrayList<>();
        this.alphaEpoch = alphaEpoch;
    }

    @Override
    public Double prepareConvergenceDataset(Double input, ML4allContext context) {
        double rmsc = 0.0;

        items.add(input);
        double alpha = (double)context.getByKey("alpha");
        long r = this.currentIteration % this.perEpochSize;
        if (this.currentIteration % (this.alphaEpoch*this.perEpochSize) == 0){
            alpha = alpha/2;
            context.put("alpha", alpha);

            System.out.println("alpha: " + alpha  + " , " + new Timestamp(new Date().getTime()));
            double currentRMSC = this.calcualteRMSE();

            System.out.println(this.currentIteration + "(" + this.currentIteration/this.perEpochSize + ") , " + currentRMSC + " , " + currentRMSC );
        }

        return rmsc;

    }

    private double calcualteRMSE(){
        double rmsc = 0.0;
        if(this.items.size() > 0){
            double s =0.0;
            for(double item : this.items){
                s = s + Math.pow(item, 2);
            }

            double s2 = s/this.items.size();

            rmsc =  Math.sqrt(s2);
            this.items.clear();
        }
        return rmsc;

    }
    @Override
    public boolean terminate(Double input) {
/*        if(input < 0.001){
            return true;
        }*/
        int current_epoch = (int)(this.currentIteration / this.perEpochSize);
        return ( ++this.currentIteration >= this.maxIternation || current_epoch >= this.epoch);
    }


}
