package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.qcri.ml4all.abstraction.plan.Platforms;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.data.Tuple2;

import java.io.IOException;
import java.math.BigDecimal;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import static org.qcri.ml4all.abstraction.plan.Platforms.SPARK_JAVA;

public class NMFTestGenerator {


    static Map<String, Integer> indexIter;

    static int datasetSize  = 5;
    static int features = 4;
    static int k = 2;

    static int max_iterations = 500;

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

    public NMFTestGenerator() {
        try {
            this.document = Nd4j.readNumpy("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/sample.txt",",");
        } catch (IOException e) {
            e.printStackTrace();
        }
        rmse=new ArrayList<Double>();
        omega = this.calculateOmega();

        indexIter = new HashMap<>();
        double max = Math.sqrt(k);

        Nd4j.getRandom().setSeed(k);
        w = Nd4j.rand(new int[]{datasetSize, k}, 0.0, max, Nd4j.getRandom());
        h = Nd4j.rand(new int[]{k, features}, 0.0, max, Nd4j.getRandom());

      //  w = Nd4j.ones(new int[]{datasetSize, k});
      //  h = Nd4j.ones(new int[]{k, features});

        System.out.println("W init : " + w);
        System.out.println("H init : " + h);
        System.out.println();
        System.out.println();
        int index = 0;
        while(index < max_iterations){
            System.out.println("Iteration : " + index);
            //this.simpleCompute();
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


        int i = this.getRandomIndexPointer(1, datasetSize -1);
        int j = this.getRandomIndexPointer(1, features - 1);
       // int i= 3;
       // int j=3;
        current_rmse = this.calcualteRMSE(i, j);
       // System.out.println("current_rmse");
        rmse.add(current_rmse);

        double aDataPoint = this.document.getDouble(i,j);

        if(aDataPoint > 0 && current_rmse > 0){
                INDArray updateW = null;
                INDArray updateH = null;

                double stepSize = this.getStepSize(i, j);

                System.out.println("w = " + w);
                System.out.println("h = " + h);
                System.out.println("w[i] = " + w.getRow(i));
                System.out.println("h[j] = " + h.getColumn(j));
                System.out.println("i,j(" + i + "," + j + ") = " + aDataPoint);
                System.out.println("stepSize = " + stepSize);

                System.out.println();
                double aW = aDataPoint - (w.getRow(i).mmul(h.getColumn(j)).getDouble(0));
                System.out.println("aW = " + aW);
                System.out.println();

                updateW = h.getColumn(j).mul(aW);
                System.out.println("1.updateW = " + updateW);
                System.out.println();

                updateW = updateW.add((w.getRow(i).mul(this.regulizer)).transpose());
                System.out.println("2.updateW = " + updateW);
                System.out.println();

                updateW = updateW.mul(stepSize);
                System.out.println("3.updateW = " + updateW);
                System.out.println();

                updateW = w.getRow(i).sub(updateW);
                System.out.println("4.updateW = " + updateW);
                System.out.println();

                double aH = aDataPoint - (updateW.mmul(h.getColumn(j)).getDouble(0));
                System.out.println("aH = " + aH);
                System.out.println();

                updateH = updateW.mul(aH);
                System.out.println("1.updateH = " + updateH);
                System.out.println();

                updateH = updateH.add((h.getColumn(j).mul(this.regulizer)).transpose());
                System.out.println("2.updateH = " + updateH);
                System.out.println();

                updateH = updateH.mul(stepSize);
                System.out.println("3.updateH = " + updateH);
                System.out.println();

                this.update(updateW, updateH, i, j);
        }

    }

    public void update(INDArray newW, INDArray newH,int i, int j) {

        if(newW != null && newH != null){

            INDArray updateW = Transforms.max(newW, lower_bound, true);
            System.out.println("updateW = " + updateW);
            System.out.println();

            INDArray updateH = h.getColumn(j).sub(newH);
            System.out.println("updateH = " + updateH);
            System.out.println();


            updateH = Transforms.max(updateH, lower_bound, true);
            System.out.println("Transforms.max(updateH, lower_bound, true) = " + Transforms.max(updateH, lower_bound, true));
            System.out.println("updateH = " + updateH);
            System.out.println();

            System.out.println("w[i] = " + w.getRow(i));
            System.out.println("h[j] = " + h.getColumn(j));

            w.putRow(i, updateW);
            h.putColumn(j, updateH);

            System.out.println("w[i] = " + w.getRow(i));
            System.out.println("h[j] = " + h.getColumn(j));
            System.out.println();
        }

    }



    private int getRandomIndexPointer(int min, int max){
        // inclusive min and max
        return min+(int)(Math.random()*((max-min ) + 1));
    }

    private double getStepSize( int i, int j){

        String key = i+","+j;
        int updatedIteration = 0;

        if(indexIter.containsKey(key)){
            updatedIteration = indexIter.get(key);
            updatedIteration++;
        }



        indexIter.put(key, updatedIteration);


        double step = this.alfa / (1+ (this.beta * Math.pow(updatedIteration, 1.5)));

        return step;
    }

    private double calcualteRMSE(int i, int j){
        double s1 = this.document.getDouble(i,j) - (w.getRow(i).mmul(h.getColumn(j))).getDouble(0);
        double s2 =(Math.pow(s1, 2))/omega;

        return Math.sqrt(s2);
    }
    private int calculateOmega(){
        int nonZeroValueCount = 0;
        for(int i =0; i < this.document.rows(); i++){
            for(int j=0; j < this.document.columns(); j++){
                if(this.document.getDouble(i,j) > 0.0){
                    nonZeroValueCount++;
                }
            }
        }

        return nonZeroValueCount;
    }


    public void simpleCompute() {


        int i = this.getRandomIndexPointer(1, datasetSize -1);
        int j = this.getRandomIndexPointer(1, features - 1);

        double aDataPoint = this.document.getDouble(i,j);

        if(aDataPoint > 0){
            // for(int a=0; a < k; a++) {
            INDArray updateW = null;
            INDArray updateH = null;

            double stepSize = this.getStepSize(i, j);
            System.out.println("w = " + w);
            System.out.println("h = " + h);
            System.out.println("w[i] = " + w.getRow(i));
            System.out.println("h[j] = " + h.getColumn(j));
            System.out.println("i,j(" + i + "," + j + ") = " + aDataPoint);
            System.out.println("stepSize = " + stepSize);

            System.out.println();

            updateW = w.getRow(i).sub((h.getColumn(j).mul(aDataPoint - (w.getRow(i).mmul(h.getColumn(j)).getDouble(0))).add((w.getRow(i).mul(this.regulizer)).transpose())).mul(stepSize));
            INDArray uWi = updateW;


            System.out.println("1.updateW = " + updateW);
            System.out.println();

            updateH = (uWi.mul(aDataPoint - (uWi.mmul(h.getColumn(j)).getDouble(0))).add((h.getColumn(j).mul(this.regulizer)).transpose())).mul(stepSize);
            System.out.println("2.updateW = " + updateH);

            System.out.println();

            this.update(updateW, updateH, i, j);
            //  }
        }

    }
}
