package org.qcri.ml4all.examples.nomad;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.SerializationUtils;
import org.qcri.ml4all.abstraction.api.LocalStage;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.ml4all.examples.util.StringUtil;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;


public class NomadStageWithRandomValues extends LocalStage {
    String fileName ;
    double min = 0.0;
    double max;
    int[] wShape;
    int[] hShape;
    int m;
    int n;

    public NomadStageWithRandomValues(int k, int m, int n, String fileName) {
        this.fileName = fileName;
        this.max = Math.sqrt(k);
        this.wShape = new int[]{m, k};
        this.hShape = new int[]{n, k};
        this.m = m;
        this.n = n;
    }

    @Override
    public void staging (ML4allContext context) {

        INDArray w = Nd4j.rand(wShape, min, max, Nd4j.getRandom());
        INDArray h = Nd4j.rand(hShape, min, max, Nd4j.getRandom());
        INDArray data = Nd4j.zeros(m,n);

        context.put("w", w);
        context.put("h", h);
        context.put("a", data);
       // context.put("a", this.convertDataToMatrix());
        context.put("iter", 1);

    }

    private INDArray convertDataToMatrix(){

        List<String> list = new ArrayList<>();
        INDArray myArray = null;

        try (Stream<String> stream = Files.lines(Paths.get(this.fileName))) {
            list = stream.collect(Collectors.toList());
            int nRow = list.size();
            int nCol = list.get(0).split(",").length;
            myArray = Nd4j.zeros(nRow, nCol);

            for(int i =0; i < list.size(); i++){
                List<String> pointStr = StringUtil.split(list.get(i),',');
                double[] point = new double[pointStr.size()];
                for(int j=0; j < pointStr.size(); j++){
                    point[j] = Double.parseDouble(pointStr.get(j));
                    myArray.putScalar(i, j, point[j]);
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e){
            e.printStackTrace();
        }
        return myArray;
    }

}
