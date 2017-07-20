package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.Compute;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class NMFCompute extends Compute<Tuple2<Tuple2, int[]>, double[]> {

    private double beta =  0.01;
    private static long itr = 1;
    private int k;

    public NMFCompute(double beta, int k) {
        this.beta = beta;
        this.k = k;
    }

    @Override
    public Tuple2 process(double[] input , ML4allContext context) {
        INDArray updateW = null;
        INDArray updateH = null;

        int[] point = new int[2];

        int i = (int)input[0];
        int j = this.getRandomIndexPointer(0, k);

        double aDataPoint = input[j];

        point[0] = i;
        point[1] = j;
       // System.out.println("i, j ==>  " + i + "," + j);
        double alpha = this.setStepSize();
        itr++;
        if(aDataPoint <= 0){
           return new Tuple2(new Tuple2(updateW, updateH), new Tuple2(point, 0.0));
        }

        INDArray w = (INDArray)context.getByKey("w");
        INDArray h = (INDArray)context.getByKey("h");

        double aW = aDataPoint - (w.getRow(i).mmul(h.getColumn(j)).getDouble(0));


        updateW = ((h.getColumn(j).mul(aW)).sub(w.getRow(i).mul(this.beta))).mul(alpha).transpose();

        updateH = ((w.getRow(i).mul(aW)).sub(h.getColumn(j).mul(this.beta))).mul(alpha).transpose();

        return new Tuple2(new Tuple2(updateW, updateH), new Tuple2(point, aW));
    }

    private double setStepSize(){
        double stepSize = (20.0 / (itr+10.0));

        return stepSize;

    }

    private int getRandomIndexPointer(int min, int max){
        return min+(int)(Math.random()*((max-min ) + 1));
    }

}
