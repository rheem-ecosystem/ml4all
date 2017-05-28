package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class NMFTestMatrixFactorization {


    static Map<String, Integer> indexIter;

    static int datasetSize  = 5;
    static int features = 4;
    static int k = 2;

    static int max_iterations = 5000;

    static double regulizer = 0.0002;
    static double lower_bound = 0.0;
    static double alfa= 0.002;
    static double beta = 0.002;


    static INDArray document;

    static INDArray w ;
    static INDArray h ;

    static int omega;
    static double current_rmse = 1.0;
    ArrayList<Double> rmse;

    public NMFTestMatrixFactorization() {
        try {
            this.document = Nd4j.readNumpy("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/sample.txt",",");
        } catch (IOException e) {
            e.printStackTrace();
        }


        indexIter = new HashMap<>();
        double max = Math.sqrt(k);

        Nd4j.getRandom().setSeed(k);
        w = Nd4j.rand(new int[]{datasetSize, k}, 0.0, max, Nd4j.getRandom());
        h = Nd4j.rand(new int[]{features, k}, 0.0, max, Nd4j.getRandom());
        h = h.transpose();

        System.out.println("W init : " + w);
        System.out.println("H init : " + h);
        System.out.println();
        System.out.println();
        int index = 0;

        while(index < max_iterations){
            System.out.println("Iteration : " + index);
            this.compute();
            index++;
        }

        System.out.println();
        System.out.println();
        System.out.println("W final : " + w);
        System.out.println("H final : " + h);
        System.out.println();
        System.out.println();
        INDArray finalOut = w.mmul(h);
        System.out.println("finalOut : " + finalOut);
       // rmse.forEach(item -> System.out.println(item));
    }

    public void compute() {

        for(int i=0; i < this.document.rows(); i++){
            for(int j=0; j< this.document.columns(); j++){
                double Rij = this.document.getDouble(i,j);
                if(Rij > 0){
                    double eij = Rij - (w.getRow(i).mmul(h.getColumn(j))).getDouble(0);
                    for(int ak=0; ak < k; ak++){

                        double newW = w.getDouble(i,ak) + alfa * (2 * eij * h.getDouble(ak,j) - beta* w.getDouble(i,ak));
                        w.putScalar(i,ak,newW);

                        double newH = h.getDouble(ak,j) + alfa * (2*eij*w.getDouble(i,ak) - beta*h.getDouble(ak,j) );
                        h.putScalar(ak,j, newH);
                    }

                }
            }
        }

    }



}
