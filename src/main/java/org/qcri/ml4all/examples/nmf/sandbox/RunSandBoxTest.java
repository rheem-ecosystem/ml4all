package org.qcri.ml4all.examples.nmf.sandbox;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.qcri.ml4all.abstraction.plan.Platforms;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

import java.io.File;
import java.net.MalformedURLException;

import static org.qcri.ml4all.abstraction.plan.Platforms.*;

/**
 * Execute SGD for logistic regression.
 */
public class RunSandBoxTest {

    // Default parameters.

    static String relativePath = "src/main/resources/input/sample.txt";
    static int datasetSize  = 5;
    static int features = 4;
    static int k = 2;

    static int max_iterations = 5;
    static Platforms platform = SPARK_JAVA;

    static double regulizer = 0.02;
    static double lower_bound = 0.0;
    static double alfa= 0.0002;
    static double beta = 0.02;
    static INDArray docData;
//alpha=0.0002, beta=0.02
    public static void main (String... args) throws MalformedURLException {

      //   w=[1 2 3] and h = [1 1 1]
       // INDArray w = Nd4j.create(new float[]{1,2,3},new int[]{1,3});
      //  INDArray h = Nd4j.create(,new int[]{3,1});
      //  System.out.println(w.mmul(h).getDouble(0));

     //   INDArray w = Nd4j.rand(new int[]{10, 3}, Nd4j.getRandom());
     //   INDArray h = Nd4j.rand(new int[]{3, 5}, Nd4j.getRandom());

     //   INDArray A = w.mmul(h);
   //     double rmsc = calcualteRMSE(w,h,A);
        //System.out.println(w.mmul(h).getDouble(0));
     //   System.out.println(rmsc);
     //   System.out.println("");



        new NMFJuliaBatchMatrixFactorization();
    }

    private static double calcualteRMSE(INDArray w, INDArray h, INDArray trainingDocument){

        int nonZeroValue = 0;
        double s =0.0;
        for(int i=0; i < trainingDocument.rows(); i++){
            for(int j=0; j < trainingDocument.columns(); j++){
                if(trainingDocument.getDouble(i,j) > 0){
                    nonZeroValue++;
                    double s1 =  trainingDocument.getDouble(i,j) - (
                            w.getRow(i).mmul(
                                    h.getColumn(j)
                            )).getDouble(0);
                    s = s + Math.pow(s1, 2);
                }
            }
        }

        double s2 = s/nonZeroValue;

        return Math.sqrt(s2);
    }

    private static void printResult(ML4allContext context, String f){
        INDArray w = (INDArray)context.getByKey("w");
        INDArray h = (INDArray)context.getByKey("h");
        System.out.println("W final : " + w);
        System.out.println("H final : " + h);
        INDArray finalOut = w.mmul(h);
        System.out.println("finalOut : " + finalOut);
        Nd4j.writeTxt(finalOut, new File(f).getName() );


    }
    private static void setClassVariables(String... args){
        if (args.length > 0) {
            relativePath = args[0];
            datasetSize = Integer.parseInt(args[1]);
            features = Integer.parseInt(args[2]);
            max_iterations = Integer.parseInt(args[3]);
            regulizer = Double.parseDouble(args[4]);
            alfa = Double.parseDouble(args[5]);
            beta = Double.parseDouble(args[6]);
            String platformIn = args[7];
            switch (platformIn) {
                case "spark":
                    platform = SPARK;
                    break;
                case "java":
                    platform = JAVA;
                    break;
                case "any":
                    platform = SPARK_JAVA;
                    break;
                default:
                    System.err.format("Unknown platform: \"%s\"\n", platform);
                    System.exit(3);
            }
        }
        else {
            System.out.println("Usage: java <main class> [<dataset path> <dataset size> <#features> <max maxIterations> <accuracy> <sample size>]");
            System.out.println("Loading default values");
        }

    }

}


