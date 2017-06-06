package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.Date;
import java.util.Map;

public class NMFTestMatrixFactorization {

    int m_datasetSize ;
    int n_features;
    int k = 5;

    static int max_iterations = 1000;

    static double alfa= 0.00001;
    static double beta = 0.001;

    static INDArray document;

    static INDArray w ;
    static INDArray h ;

    String outputPath = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/out/";
    public NMFTestMatrixFactorization() {

        this.init();
       // int mn = Math.min(m_datasetSize, n_features);

        System.out.println("W init : " + w);
        System.out.println("H init : " + h);
        System.out.println();
        int index = 0;

        while(index < max_iterations){
            this.compute();
            index++;
            System.out.println(index);
        }

        System.out.println();
        System.out.println("W final : " + w);
        System.out.println();
        System.out.println("H final : " + h);

       // System.out.println("R final : " + w.mmul(h));

        generateOutput();
    }

    public void init(){
        try {
            //document = Nd4j.readNumpy("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/sample.txt",",");
            document = Nd4j.readNumpy("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apgRandomTest.txt",",");
            m_datasetSize = document.rows();
            n_features = document.columns();

            double max = Math.sqrt(k);

            //Nd4j.getRandom().setSeed(k);
            w = Nd4j.rand(new int[]{m_datasetSize, k}, 0.0, max, Nd4j.getRandom());
            h = Nd4j.rand(new int[]{n_features, k}, 0.0, max, Nd4j.getRandom());
            h = h.transpose();


        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void generateOutput(){

        // rmse.forEach(item -> System.out.println(item));
        try{

            File fs = new File(outputPath +"apg_cell_w_"+ new Date().getTime() + ".bin");
            Nd4j.saveBinary(w, fs);

            File fs2 = new File(outputPath +"apg_cell_h1_"+ new Date().getTime() + ".bin");
            Nd4j.saveBinary(h, fs2);


            File fs3 = new File(outputPath +"apg_cell_R_"+ new Date().getTime() + ".bin");
            Nd4j.saveBinary(w.mmul(h), fs3);

            File fs4 = new File(outputPath +"apg_cell_h2_"+ new Date().getTime() + ".bin");
            Nd4j.saveBinary(h.transpose(), fs4);

        }
        catch(Exception e){
            System.out.println(e);
        }

    }

    public void compute() {
        for(int i=0; i < document.rows(); i++){
            for(int j=0; j< document.columns(); j++){
                double Rij = document.getDouble(i,j);
                if(Rij > 0){
                    double eij = Rij - (w.getRow(i).mmul(h.getColumn(j))).getDouble(0, 0);
                    for(int ak=0; ak < k; ak++){

                        double newW = w.getDouble(i,ak) + alfa * (2 * eij * h.getDouble(ak,j) - beta* w.getDouble(i,ak));
                        w.putScalar(i,ak,Math.max(0.0, newW));

                        double newH = h.getDouble(ak,j) + alfa * (2*eij*w.getDouble(i,ak) - beta*h.getDouble(ak,j) );
                        h.putScalar(ak,j, Math.max(0.0, newH));
                    }

                }
            }
        }
    }

    /**
    public boolean calculateError(){
        boolean stopCompute = false;
        double e = 0;
        for(int i=0; i < document.rows(); i++) {
            for(int j=0; j< document.columns(); j++) {
                double Rij = document.getDouble(i,j);
                if(Rij > 0){
                    double aValue = w.getRow(i).mmul(h.getColumn(j)).getDouble(0);
                    e = e + Math.pow(Rij - aValue,2);
                    for(int ak=0; ak < k; ak++) {
                        e = e + (regulizer / 2) * (Math.pow(w.getDouble(i, ak), 2) + Math.pow(h.getDouble(ak,j), 2));
                    }
                }
            }
            if(e < 0.001){
                stopCompute = true;
                break;
            }
        }
        System.out.println( "ERROR Score: " + e);
        return stopCompute;
    }

    **/

}
