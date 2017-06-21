package org.qcri.ml4all.examples.nmf.sandbox;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by jlucas on 5/28/17.
 */
public class NetFlixMVPreprocessor {

    private static String outputPath = "/Users/jlucas/Documents/Rheem/data/netflix/output/";
    public static void main (String... args) throws MalformedURLException {
        String path = "/Users/jlucas/Documents/Rheem/data/netflix/download/training_set";


        File directory = new File(path);

        File[] fList = directory.listFiles();

        for (File file : fList){
            if (!file.isDirectory()){
                List<String> list = new ArrayList<>();

                try (BufferedReader br = Files.newBufferedReader(Paths.get(file.getAbsolutePath()))) {

                    list = br.lines().collect(Collectors.toList());

                } catch (IOException e) {
                    e.printStackTrace();
                }

                generateTransformatedData(list);

            }
        }

    }

    private static void generateTransformatedData(List<String> list){
        //userid, movieid, rating
        String header = list.get(0).replace(":", "");
        StringBuffer sb = new StringBuffer();
        for(int i=1; i < list.size(); i++){
            String input[] = list.get(i).split(",");
            sb.append(input[0]);
            sb.append(",");
            sb.append(header);
            sb.append(",");
            sb.append(input[1]);
            sb.append(System.lineSeparator());
        }

        String oPath = outputPath + header + ".txt";
        try {
          //  Files.write(Paths.get(oPath),
            //        sb.toString().getBytes(),
             //       StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            Files.write(Paths.get(oPath), sb.toString().getBytes());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}
