package org.qcri.ml4all.examples.nmf;

import org.apache.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.qcri.ml4all.abstraction.api.LocalStage;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;


public class NMFStageWithRandomValues extends LocalStage {

    private static Logger logger = Logger.getLogger(NMFStageWithRandomValues.class);

    int[] wShape;
    int[] hShape;
    int seed;
    double avg;
    String wPath;
    String hPath;
    boolean basedSeedVector = false;

    public NMFStageWithRandomValues(int k, int seed, int m, int n, double avg, String wPath, String hPath, boolean basedSeedVector) {

        this.wShape = new int[]{m, k};
        this.hShape = new int[]{n, k};
        this.seed = seed;
        this.avg = avg;
        if(basedSeedVector) {
            this.basedSeedVector = basedSeedVector;
            this.wPath = wPath;
            this.hPath = hPath;
        }
    }

    @Override
    public void staging (ML4allContext context) {
        INDArray w = null;
        INDArray h = null;
        if(this.basedSeedVector){
            try{
                w = Nd4j.readNumpy(this.wPath, ",");
                h = Nd4j.readNumpy(this.hPath, ",");
            }
            catch (Exception e){
                System.out.println(e);
                logger.error(e);
            }
        }
        else{
            org.nd4j.linalg.api.rng.Random rnd = Nd4j.getRandom();
            rnd.setSeed(this.seed);

            w = Nd4j.randn(this.wShape ,  rnd);
            w = w.mul(this.avg);
            w = Transforms.abs(w);

            h = Nd4j.randn(this.hShape, rnd);
            h = h.mul(avg);
            h = Transforms.abs(h);
            h = h.transpose();
        }
        context.put("w", w);
        context.put("h", h);
    }

}
