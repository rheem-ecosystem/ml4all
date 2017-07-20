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

    long maxIterations;
    long currentIteration = 1;
    double minRMSC;
    int epoch;
    List<Double> items;
    int per_epoch_size ;

    public NMFLoop(int datasetSize, int features, double minRMSC, int epoch) {

        this.minRMSC = minRMSC;
        this.epoch = epoch;
        this.items = new ArrayList<>();
        this.per_epoch_size = datasetSize*features;
        this.maxIterations =(datasetSize*features) * (long)epoch;
        System.out.println("===  " + maxIterations + " , " + this.epoch + " , " + this.per_epoch_size + "   ===  ");
    }

    @Override
    public Double prepareConvergenceDataset(Double input, ML4allContext context) {
        double rmsc = 0.0;
       // System.out.println(this.currentIteration );

        if(input > 0.0){
            items.add(input);
        }

        int k = (int)((this.currentIteration) % this.per_epoch_size);
       // System.out.println("prepareConvergenceDataset k : " + k);
/*
        if((currentIteration) == 26341991){
            System.out.println("prepareConvergenceDataset : " + currentIteration);
        }
*/

        if(k == 0 ) {
            System.out.println("items size : " + this.items.size() + " , " + new Timestamp(new Date().getTime()) );
            System.out.println("epoch size : " + ((this.currentIteration) / this.per_epoch_size) + " , " + this.currentIteration);
            if(this.items.size() > 0){
                double s =0.0;
                for(double item : this.items){
                    s = s + Math.pow(item, 2);
                }

                double s2 = s/this.items.size();

                rmsc =  Math.sqrt(s2);
                this.items.clear();

               }
            System.out.println("==============>>>>>>>>> " + new Timestamp(new Date().getTime())+ " , " + this.currentIteration + " , " + rmsc);
        }


        return rmsc;

    }

    @Override
    public boolean terminate(Double input) {
        int current_epoch = (int)(this.currentIteration / this.per_epoch_size);
        return ( ++this.currentIteration >= this.maxIterations || current_epoch >= this.epoch);
    }


}
