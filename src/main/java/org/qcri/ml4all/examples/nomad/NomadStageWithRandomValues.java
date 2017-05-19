package org.qcri.ml4all.examples.nomad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.qcri.ml4all.abstraction.api.LocalStage;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;



public class NomadStageWithRandomValues extends LocalStage {
    double min = 0.0;
    double max;
    int[] wShape;
    int[] hShape;
    int m;
    int n;

    public NomadStageWithRandomValues(int k, int m, int n) {
        this.max = Math.sqrt(k);
        this.wShape = new int[]{m, k};
        this.hShape = new int[]{k, n};
        this.m = m;
        this.n = n;
    }

    @Override
    public void staging (ML4allContext context) {

        INDArray w = Nd4j.rand(wShape, min, max, Nd4j.getRandom());
        INDArray h = Nd4j.rand(hShape, min, max, Nd4j.getRandom());

        context.put("w", w);
        context.put("h", h);
        context.put("m", m);
        context.put("n", n);
        context.put("iter", 1);

    }

}
