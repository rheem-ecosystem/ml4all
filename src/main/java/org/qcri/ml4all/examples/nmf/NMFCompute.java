package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.qcri.ml4all.abstraction.api.Compute;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

import java.util.HashMap;
import java.util.Random;

public class NMFCompute extends Compute<Tuple2, double[]> {

    double regulizer = 1.0;
    int m;
    int n;
    double alfa;
    double beta;
    Random rand;
    public NMFCompute(double regulizer, int m, int n, double alfa, double beta) {
        this.regulizer = regulizer;
        this.m = m;
        this.n = n;
        this.alfa = alfa;
        this.beta = beta;

        this.rand = new Random();
    }

    @Override
    public Tuple2 process(double[] input , ML4allContext context) {

        INDArray w = (INDArray) context.getByKey("w");
        INDArray h = (INDArray) context.getByKey("h");

        int i = (int)input[0];
        int j = getRandomIndexPointer(1, this.n -1);

        System.out.println("i : " + i + "   ========   j: " + j);
        double stepSize = this.getStepSize(context, i, j);

        double aDataPoint = input[j];

        double aW = aDataPoint - (w.getRow(i).mmul(h.getColumn(j)).getDouble(0));
        System.out.println("inner project: " + w.getRow(i).mmul(h.getColumn(j)));

        INDArray updateW =h.getColumn(j).mul(aW).transpose();
        updateW = updateW.add(w.getRow(i).mul(regulizer));
        updateW = updateW.mul(stepSize);


        double aH = aDataPoint - (w.getRow(i).mmul(h.getColumn(j)).getDouble(0)) ;
        System.out.println("inner project: " + w.getRow(i).mmul(h.getColumn(j)));

        INDArray updateH =w.getRow(i).mul(aH).transpose();
        updateH = updateH.add( h.getColumn(j).mul(regulizer));
        updateH = updateH.mul(stepSize);

        context.put("i",i);
        context.put("j",j);

        return new Tuple2(updateW, updateH);
    }


    private int getRandomIndexPointer(int min, int max){
        // inclusive min and max
        return min+(int)(Math.random()*((max-min ) + 1));
    }
    private double getStepSize(ML4allContext context, int i, int j){
        HashMap<String, Integer>indexIter =  (HashMap)context.getByKey("indexIter");
        String key = i+","+j;
        int updatedIteration = 0;

        if(indexIter.containsKey(key)){
            updatedIteration = indexIter.get(key);
        }
        updatedIteration++;

        indexIter.put(key, updatedIteration);

        return this.alfa / (1+ (this.beta * Math.pow(updatedIteration, 1.5))) ;
    }
}
