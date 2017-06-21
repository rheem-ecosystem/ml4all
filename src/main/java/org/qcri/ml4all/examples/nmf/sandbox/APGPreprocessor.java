package org.qcri.ml4all.examples.nmf.sandbox;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Created by jlucas on 5/28/17.
 */
public class APGPreprocessor {

    private static String outputPath = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/";
    public static void main (String... args) throws MalformedURLException {

        try {
            //filterFeatures();
            //filterUsers();
            postProcess();

        } catch (Exception e) {
            e.printStackTrace();
        }


/**
        String path = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/aje-youtube-big-table.csv";
        List<String> list = new ArrayList<>();

        try (BufferedReader br = Files.newBufferedReader(Paths.get(path))) {

            list = br.lines().collect(Collectors.toList());

        } catch (IOException e) {
            e.printStackTrace();
        }

        generateTransformatedData(list);
 **/

    }

    private static void generateTransformatedData(List<String> list){
        //userid, movieid, rating
        list.remove(0);
        StringBuffer sb = new StringBuffer();
        for(int i=1; i < list.size(); i++){
            String input[] = list.get(i).split(",");
           // System.out.println(list.get(i).toString());
            for(int j=1; j < input.length; j++){
                if(isNumeric(input[j])){
                    sb.append(input[j]);
                    if( j < (input.length - 1)){
                        sb.append(",");
                    }
                }
                else{
                    System.out.println(list.get(i));
                }
            }

            sb.append("\r");
           // System.out.println(sb.toString());
        }

        String oPath = outputPath + "apgNoHeaderInfoBigTable.txt";
        try {

            Files.write(Paths.get(oPath), sb.toString().getBytes());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static boolean isNumeric(String o){
        try{
            Double.valueOf(o);
            return true;
        }
        catch(Exception e){
            System.out.println(o);
           return false;
        }

    }

    private static void filterFeatures() throws IOException {
        INDArray matrix = Nd4j.readNumpy("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apgNoHeaderInfoBigTable.txt",",");

        Map<Integer, APGFeature> items = new HashMap<>();
        String outputPath = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/";

        //   min[sum[users[i,:]]]
        for(int i =0; i < matrix.columns(); i++){
            double feasureSum = matrix.getColumn(i).sumNumber().doubleValue();
            if(feasureSum > 0.0){
                Map<Integer, Double> nonZeros = new HashMap<>();
                INDArray col = matrix.getColumn(i);
                int countNonCol = 0;
                for(int j=0; j < col.rows(); j++){
                    if(col.getDouble(j,0) > 0){
                        countNonCol++;
                        nonZeros.put(j, col.getDouble(j,0));
                    }
                }
                APGFeature apgFeature = new APGFeature(feasureSum, countNonCol, nonZeros);
                items.put(i, apgFeature);
            }
        }

        INDArray newDoc = null;
        for(int i =0; i < matrix.columns(); i++){
            if(items.containsKey(new Integer(i))){
                if(newDoc == null){
                    newDoc = matrix.getColumn(i);
                }
                else{
                    Nd4j.vstack(newDoc, matrix.getColumn(i));
                }
            }
        }
        System.out.println(newDoc.shapeInfoToString());
        File fs4 = new File(outputPath +"apg_filter_batch.bin");
        Nd4j.saveBinary(newDoc, fs4);

    }

    private static void filterUsers() throws IOException {
        INDArray matrix = Nd4j.readNumpy("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apgNoHeaderInfoBigTable.txt",",");

        Map<Integer, APGFeature> items = new HashMap<>();
        String outputPath = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/";

        //   min[sum[users[i,:]]]
        for(int i =0; i < matrix.rows(); i++){
           // double feasureSum = matrix.getRow(i).sumNumber().doubleValue();
           // if(feasureSum > 0.0){
                double sum = 0;
                INDArray aRow = matrix.getRow(i);
                int countNonCol = 0;
                Map<Integer, Double> nonZeros = new HashMap<>();
                for(int j=0; j < aRow.columns(); j++){
                    if(aRow.getDouble(0,j) > 0.0){
                        countNonCol++;
                        sum = sum + aRow.getDouble(0,j);
                        nonZeros.put(j, aRow.getDouble(0,j));
                    }
                }
                if(countNonCol >= 300){
                    APGFeature apgFeature = new APGFeature(sum, countNonCol, nonZeros);
                    items.put(i, apgFeature);
                    nonZeros = null;
                }

            //}
        }

        INDArray newDoc = null;
        for(int i =0; i < matrix.rows(); i++){
            if(items.containsKey(new Integer(i))){
                if(newDoc == null){
                    newDoc = matrix.getRow(i);
                }
                else{
                    newDoc = Nd4j.vstack(newDoc, matrix.getRow(i));
                }
            }
        }
        System.out.println(newDoc.shapeInfoToString());
        File fs4 = new File(outputPath +"apg_filter_batch_200_f.bin");
        Nd4j.saveBinary(newDoc, fs4);

    }

    public static void postProcess() throws Exception{
        ///Users/jlucas/Documents/Rheem/ml4all/src/main/resources/out/apg_Julia_300.bin
        int k = 5;
        Map<Integer, APGFeature> items = getMatrixInfo();

        INDArray R =  Nd4j.readBinary(new File("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/out/apg_Julia_300.bin"));
        generateCompleteOuputFile(R, items , "R");

        INDArray H =  Nd4j.readBinary(new File("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/out/apg_Julia_300_h.bin"));

        generateCompleteOuputFile(H, items , "H");
      /**
       * INDArray R =  Nd4j.readBinary(new File("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/out/apg_Julia_300.bin"));

       INDArray W =  Nd4j.readBinary(new File("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/out/apg_Julia_300_w.bin"));
        INDArray H =  Nd4j.readBinary(new File("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/out/apg_Julia_300_h.bin"));
       **/

    }

    public static Map<Integer, APGFeature> getMatrixInfo() throws Exception{
        INDArray matrix = Nd4j.readNumpy("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apgNoHeaderInfoBigTable.txt",",");

        Map<Integer, APGFeature> items = new HashMap<>();
        String outputPath = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/";

        //   min[sum[users[i,:]]]
        for(int i =0; i < matrix.rows(); i++){
            // double feasureSum = matrix.getRow(i).sumNumber().doubleValue();
            // if(feasureSum > 0.0){
            double sum = 0;
            INDArray aRow = matrix.getRow(i);
            int countNonCol = 0;
            Map<Integer, Double> nonZeros = new HashMap<>();
            for(int j=0; j < aRow.columns(); j++){
                if(aRow.getDouble(0,j) > 0.0){
                    countNonCol++;
                    sum = sum + aRow.getDouble(0,j);
                    nonZeros.put(j, aRow.getDouble(0,j));
                }
            }
            if(countNonCol >= 300){
                APGFeature apgFeature = new APGFeature(sum, countNonCol, nonZeros);
                items.put(i, apgFeature);
                nonZeros = null;
            }
        }

        return items;
    }

    public static void generateCompleteOuputFile(INDArray aMatrix, Map<Integer, APGFeature> items, String RfILE){
        String path = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/aje-youtube-big-table.csv";
        List<String> list = new ArrayList<>();

        try (BufferedReader br = Files.newBufferedReader(Paths.get(path))) {

            list = br.lines().collect(Collectors.toList());

        } catch (IOException e) {
            e.printStackTrace();
        }


        Map<Integer, String> users = new HashMap<>();
        int userIndex = 0;
        for(int i=1; i < list.size(); i++){
            String input[] = list.get(i).split(",");
            if(items.containsKey(new Integer(i))){
                users.put(userIndex, input[0]);
                userIndex++;
            }

        }


        String headers = list.get(0);
        StringBuffer sb = new StringBuffer();
        sb.append(headers);
       // sb.append(System.lineSeparator());
        sb.append("\r");
        for(int aIndex = 0; aIndex < aMatrix.rows(); aIndex++){
            System.out.println(aIndex);
           // if(items.containsKey(new Integer(aIndex))){
                String aRow = aMatrix.getRow(aIndex).toString().replace("[","").replace("]","");
                if(RfILE.equalsIgnoreCase("R")) {
                    aRow = users.get(aIndex) + "," + aRow;
                }
                else {
                    aRow = " ," + aRow;
                }
                sb.append(aRow);
                sb.append("\r");
              //  sb.append(System.lineSeparator());
          //  }

        }

        String oPathParent = "/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/out/";
        String oPath = oPathParent + "apg_300_post_" + RfILE +".csv";
        try {

            Files.write(Paths.get(oPath), sb.toString().getBytes());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
