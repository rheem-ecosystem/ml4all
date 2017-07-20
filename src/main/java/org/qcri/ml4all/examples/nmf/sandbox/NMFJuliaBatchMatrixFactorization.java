package org.qcri.ml4all.examples.nmf.sandbox;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.io.IOException;
import java.sql.Timestamp;
import java.util.*;

public class NMFJuliaBatchMatrixFactorization {

    static Map<String, Integer> indexIter;
    static int datasetSize;
    static int features;
    static int k = 5;

    static double beta = 0.01;
    static double lower_bound = 0.0;
    static int omegaTest;

    static INDArray w ;
    static INDArray h ;

    static INDArray trainingDocument;


    String outputPath = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/out/";

    long index = 0;
    long iter = 0;
    List<Double> items = new ArrayList<>();
    public NMFJuliaBatchMatrixFactorization() {
        this.init();

        System.out.println(new Timestamp(new Date().getTime()));
        long pointR =0;
        int max_epoch =500;
        while(pointR < max_epoch){

            this.compute();
           // System.out.println(index);
            int r = (int)(index % (trainingDocument.rows() *trainingDocument.columns()));
            pointR = index / (trainingDocument.rows() *trainingDocument.columns());
            if(r == 0){
                System.out.println(new Timestamp(new Date().getTime()));
                double currentRMSC = this.calcualteRMSE();
              //  if(currentRMSC > 0.0) {
                    System.out.println(index + "(" + pointR + ") , " + currentRMSC);
             //   }

            }
        }
        System.out.println(new Timestamp(new Date().getTime()));
        System.out.println("Final RMSE:=============>>>>>>>> " + this.calcualteRMSE() );

        //this.generateOutput();


    }

    private void init(){
        try {
            INDArray documentMaster = Nd4j.readNumpy("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apg_big_table.csv",",");
            //this.trainingDocument  = documentMaster.get(NDArrayIndex.interval(0, 1000), NDArrayIndex.interval(0, 100));

           // INDArray documentMaster  = Nd4j.readBinary(new File("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apg_filter_batch_300_f.bin"));
            this.trainingDocument = documentMaster;

            this.datasetSize = trainingDocument.rows();
            this.features = trainingDocument.columns();

            this.indexIter = new HashMap<>();


            double max = (1.0 / Math.sqrt(this.k));

            Nd4j.getRandom().setSeed(1234);
            this.w = Nd4j.rand(new int[]{this.datasetSize, this.k}, 0.0, max, Nd4j.getRandom());
            this.h = Nd4j.rand(new int[]{this.features,this.k}, 0.0, max, Nd4j.getRandom());
            this.h = this.h.transpose();
           // System.out.println(w);
           // System.out.println(h);

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
    private void generateOutput(){

       // rmse.forEach(item -> System.out.println(item));
        //System.out.println(w.mmul(h));
        try{
            INDArray R = w.mmul(h);
            System.out.println("trainingDocument : " + trainingDocument.getDouble(2347,0));
            System.out.println("R : " + R.getDouble(2347,0));
            System.out.println(trainingDocument.getRow(0));
            System.out.println(R.getRow(0));
            File fs4 = new File(outputPath +"apg_Julia_1000_f_r.bin");
            File fs4w = new File(outputPath +"apg_Julia_1000_f_w.bin");
            File fs4h = new File(outputPath +"apg_Julia_1000_f_h.bin");

            Nd4j.saveBinary(R, fs4);
            Nd4j.saveBinary(w, fs4w);
            Nd4j.saveBinary(h, fs4h);
        }
        catch(Exception e){
            System.out.println(e);
        }

    }

    public void compute() {
        INDArray updateW = null;
        INDArray updateH = null;


        int i = this.getRandomIndexPointer(0, this.datasetSize -1);
        int j = this.getRandomIndexPointer(0, this.features - 1);
        double aDataPoint = this.trainingDocument.getDouble(i,j);

        iter++;
        index++;
      //  System.out.println(iter + ", " + alpha);
        if(aDataPoint <= 0){
            return;
        }

        double alpha = 20.0 / (iter+10.0);
       // double alpha = 0.022/iter;

        double aW = aDataPoint - (w.getRow(i).mmul(h.getColumn(j)).getDouble(0));
        items.add(aW);

        updateW = ((h.getColumn(j).mul(aW)).sub(w.getRow(i).mul(beta))).mul(alpha).transpose();
        updateH = ((w.getRow(i).mul(aW)).sub(h.getColumn(j).mul(beta))).mul(alpha).transpose();

        this.update(updateW, updateH, i, j);

    }

    private void update(INDArray aW, INDArray aH, int i, int j){
        INDArray updateW = w.getRow(i).add(aW);
        INDArray updateH = h.getColumn(j).add(aH);
        updateW = Transforms.max(updateW, lower_bound, true);
        updateH = Transforms.max(updateH, lower_bound, true);

        w.putRow(i, updateW);
        h.putColumn(j, updateH);
    }

    private double calcualteRMSE(){
      //  System.out.println("items size : " + items.size());
        if(items.size() > 0){
            double s =0.0;
            for(double item : items){
                s = s + Math.pow(item, 2);
            }

            double s2 = s/items.size();
            items.clear();
            return Math.sqrt(s2);
        }

        return 0.0;
    }

    private double calcualteRMSE(int p){

        int nonZeroValue = 0;
        double s =0.0;
        for(int i=0; i < this.trainingDocument.rows(); i++){
            for(int j=0; j < this.trainingDocument.columns(); j++){
                if(this.trainingDocument.getDouble(i,j) > 0){
                    nonZeroValue++;
                    double s1 =  this.trainingDocument.getDouble(i,j) - (
                                            w.getRow(i).mmul(
                                                    h.getColumn(j)
                                            )).getDouble(0);
                    s = s + Math.pow(s1, 2);
                }
            }
        }
        if(this.omegaTest == 0){
            this.omegaTest = nonZeroValue;
            System.out.println("nonZeroValue:" + nonZeroValue);
        }
        double s2 = s/this.omegaTest;

        return Math.sqrt(s2);
    }

    private int getRandomIndexPointer(int min, int max){
        // inclusive min and max
        return min+(int)(Math.random()*((max-min ) + 1));
    }


}
