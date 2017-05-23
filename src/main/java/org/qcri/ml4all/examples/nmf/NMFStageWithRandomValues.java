package org.qcri.ml4all.examples.nmf;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.qcri.ml4all.abstraction.api.LocalStage;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

import java.util.HashMap;
import java.util.Map;


public class NMFStageWithRandomValues extends LocalStage {
    double min = 0.0;
    double max;
    int[] wShape;
    int[] hShape;
    Map<String, Integer> indexIter;
    public NMFStageWithRandomValues(int k, int m, int n) {
        this.max = Math.sqrt(k);
        this.wShape = new int[]{m, k};
        this.hShape = new int[]{k, n};
        this.indexIter = new HashMap<>();
    }

    @Override
    public void staging (ML4allContext context) {

        INDArray w = Nd4j.rand(wShape, min, max, Nd4j.getRandom());
        INDArray h = Nd4j.rand(hShape, min, max, Nd4j.getRandom());

        context.put("w", w);
        context.put("h", h);;
        context.put("iter", 1);
        context.put("indexIter", indexIter);

    }

}
