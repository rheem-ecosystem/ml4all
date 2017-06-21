package org.qcri.ml4all.examples.nmf.sandbox;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by jlucas on 5/30/17.
 */
public class InspectMatrix {

    public static void main (String... args) throws IOException {

        ///Users/jlucas/Documents/Rheem/ml4all/src/main/resources/out/apg_batch_R1496664036395.bin

       // generateRandomMatrix();

        ///Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apg_filter_batch_300_f.bin
        File fs = new File("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apg_filter_batch_300_f.bin");

        INDArray out = Nd4j.readBinary(fs);

       // INDArray org = Nd4j.readNumpy("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apgRandomTest.txt",",");

        System.out.println(out);

        System.out.println("test");
        System.out.println("test");


    }

    private static void generateRandomMatrix() throws IOException {
        File fs = new File("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/out/apg_batch_w1496613225963.bin");

        INDArray document = Nd4j.readBinary(fs);
        int row = 2509;
        int col = 10499;


        INDArray newDoc = Nd4j.rand(new int[]{row, col}, 0.0, 5, Nd4j.getRandom());

        for(int i=0; i < newDoc.rows(); i++){
            int k = i % 50;

            if(i == 0 || k == 0){
                newDoc.getRow(i).muli(getRandomIndexPointer(0, 3));
            }
        }

        newDoc.subi(2);
        Transforms.max(newDoc, 0.0, false);

        File fs2 = new File("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apgRandom.bin");

        Nd4j.saveBinary(newDoc, fs2);

        generateTextFile(newDoc);
    }

    private static void generateTextFile(INDArray doc) throws IOException {

        List<String> list = new ArrayList<>();
        for(int i=0; i < doc.rows(); i++){
            String a = doc.getRow(i).toString().replace("[","").replace("]","");
            System.out.println(a);
            list.add(a);
        }


        StringBuffer sb = new StringBuffer();
        for(int i=0; i < list.size(); i++){
            String input[] = list.get(i).split(",");
            for(int j=0; j < input.length; j++){
                sb.append(input[j]);
                if( j < (input.length - 1)){
                    sb.append(",");
                }

            }
            sb.append("\r");
        }

        String oPath = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apgRandomTest.txt";

        try {

            Files.write(Paths.get(oPath), sb.toString().getBytes());

        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    private static void generateTextFileFromFile() throws IOException {
        File fs = new File("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apgRandom.bin");

        INDArray doc = Nd4j.readBinary(fs);

         List<String> list = new ArrayList<>();
         for(int i=0; i < doc.rows(); i++){
         String a = doc.getRow(i).toString().replace("[","").replace("]","");
         System.out.println(a);
         list.add(a);
         }


         StringBuffer sb = new StringBuffer();
         for(int i=0; i < list.size(); i++){
             String input[] = list.get(i).split(",");
             for(int j=0; j < input.length; j++){
                 sb.append(input[j]);
                 if( j < (input.length - 1)){
                 sb.append(",");
                 }

             }
            sb.append("\r");
         }

         String oPath = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apgRandomTest.txt";
         try {

            Files.write(Paths.get(oPath), sb.toString().getBytes());

         } catch (IOException e) {
            e.printStackTrace();
         }


    }

    private static int getRandomIndexPointer(int min, int max){
        // inclusive min and max
        return min+(int)(Math.random()*((max-min ) + 1));
    }
}
