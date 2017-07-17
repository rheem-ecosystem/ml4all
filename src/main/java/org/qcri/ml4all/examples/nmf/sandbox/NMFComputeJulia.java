package org.qcri.ml4all.examples.nmf.sandbox;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.qcri.ml4all.abstraction.api.Compute;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

import java.util.HashMap;
import java.util.Map;


public class NMFComputeJulia extends Compute<Tuple2<Tuple2, int[]>, double[]> {

    private double alpha;
    private double beta;

    private static Map<String, Integer> indexIter;


    public NMFComputeJulia(double alpha, double beta) {
        this.alpha = alpha;
        this.beta = beta;

        indexIter = new HashMap<>();

    }

    @Override
    public Tuple2 process(double[] input , ML4allContext context) {
        INDArray updateW = null;
        INDArray updateH = null;

        int[] point = new int[2];

        int i = (int)input[0];
        int j = this.getRandomIndexPointer(1, input.length);

        double aDataPoint = input[j];
        point[0] = i;
        point[1] = j;

        if(aDataPoint <= 0){
            return new Tuple2(new Tuple2(updateW, updateH), point);
        }

        this.setStepSize(context);

        INDArray w = (INDArray) context.getByKey("w");
        INDArray h = (INDArray) context.getByKey("h");

        double aW = aDataPoint - (w.getRow(i).mmul(h.getColumn(j)).getDouble(0));

        updateW = (
                        (h.getColumn(j).mul(aW)).sub(w.getRow(i).mul(beta))
                ).mul(alpha).transpose();


        updateH = ((w.getRow(i).mul(aW)).sub(h.getColumn(j).mul(beta))).mul(alpha).transpose();


        return new Tuple2(new Tuple2(updateW, updateH), point);
    }

    private void setStepSize( ML4allContext context){

        int itr = (int) context.getByKey("itr");
        itr = itr + 1;

        this.alpha = 20.0 / (itr+10.0);

    }

    private int getRandomIndexPointer(int min, int max){
        return min+(int)(Math.random()*((max-min ) + 1));
    }



}
