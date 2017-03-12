package org.qcri.ml4all.abstraction.plan.wrappers;

import org.qcri.ml4all.abstraction.api.SampleSize;

import java.util.function.IntUnaryOperator;

/**
 * Created by zoi on 25/1/15.
 */
public class SampleSizeWrapper implements IntUnaryOperator {

    SampleSize sampleSizeOp;

    public SampleSizeWrapper(SampleSize sampleSizeOp) { this.sampleSizeOp = sampleSizeOp; }

    @Override
    public int applyAsInt(int iteration) {
        return (int) this.sampleSizeOp.getSampleSize(iteration);
    }
}
