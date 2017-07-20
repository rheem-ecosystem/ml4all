package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.qcri.ml4all.abstraction.api.LocalStage;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;


public class NMFStageWithRandomValues extends LocalStage {
    double min = 0.0;
    double max;
    int[] wShape;
    int[] hShape;
    int seed;

    INDArray trainingSet;

    public NMFStageWithRandomValues(int k, String source, char delimiter, int seed) {
        if(source.toLowerCase().endsWith(".bin")){
            this.loadBinarySourceData(source);
        }
        else{
            this.loadSourceData(source, delimiter);
        }

        int m = trainingSet.rows();
        int n = trainingSet.columns();
        this.max =(1.0 / Math.sqrt(k));
        this.wShape = new int[]{m, k};
        this.hShape = new int[]{n, k};
        this.seed = seed;

    }

    public NMFStageWithRandomValues(int k, int seed, int m, int n) {

        this.wShape = new int[]{m, k};
        this.hShape = new int[]{n, k};
        this.seed = seed;
        this.max =(1.0 / Math.sqrt(k));
    }

    @Override
    public void staging (ML4allContext context) {
        Nd4j.getRandom().setSeed(this.seed);
        INDArray w = Nd4j.rand(this.wShape, this.min, this.max, Nd4j.getRandom());
        INDArray h = Nd4j.rand(this.hShape, this.min, this.max, Nd4j.getRandom());
        h = h.transpose();

        context.put("w", w);
        context.put("h", h);
    }



    private void loadSourceData(String path, char delimiter) {
        try {
            this.trainingSet = Nd4j.readNumpy(path, String.valueOf(delimiter));
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private void loadBinarySourceData(String path) {
        try {
           // INDArray documentMaster  = Nd4j.readBinary
            // (new File("/Users/jlucas/Documents/Rheem/ml4all/src/main/resources/input/apg_filter_batch_300_f.bin"));

            this.trainingSet = Nd4j.readBinary(new File(path));
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
