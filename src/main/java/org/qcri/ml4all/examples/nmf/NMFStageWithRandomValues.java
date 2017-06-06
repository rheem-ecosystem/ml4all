package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.qcri.ml4all.abstraction.api.LocalStage;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;


public class NMFStageWithRandomValues extends LocalStage {
    double min = 0.0;
    double max;
    int[] wShape;
    int[] hShape;
    INDArray testSet;
    INDArray trainingSet;

    public NMFStageWithRandomValues(int k, String source, char delimiter) {
        this.loadSourceData(source, delimiter);
        int m = trainingSet.rows();
        int n = trainingSet.columns();
        this.max = Math.sqrt(k);
        this.wShape = new int[]{m, k};
        this.hShape = new int[]{k, n};

    }

    @Override
    public void staging (ML4allContext context) {

        INDArray w = Nd4j.rand(wShape, min, max, Nd4j.getRandom());
        INDArray h = Nd4j.rand(hShape, min, max, Nd4j.getRandom());

        context.put("w", w);
        context.put("h", h);;
        context.put("iter", 1);
        context.put("testSet", this.testSet);
    }



    private void loadSourceData(String path, char delimiter) {
        INDArray docData = null;
        try {
            docData = Nd4j.readNumpy(path, String.valueOf(delimiter));
            Nd4j.shuffle(docData, new Random(123), 1);

            int count = (int) (docData.rows() * 0.8);

            this.trainingSet = docData;
            this.testSet = docData.get(NDArrayIndex.interval(count, docData.rows() - 1), NDArrayIndex.all());
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
