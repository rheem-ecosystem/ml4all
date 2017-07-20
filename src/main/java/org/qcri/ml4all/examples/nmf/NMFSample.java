package org.qcri.ml4all.examples.nmf;

import org.qcri.ml4all.abstraction.api.Sample;
import org.qcri.rheem.basic.operators.SampleOperator;

public class NMFSample extends Sample {

    int sample_size;

    public NMFSample(int sample_size) {
        this.sample_size = sample_size;
    }

    @Override
    public int sampleSize() {
        return this.sample_size;
    }

    @Override
    public SampleOperator.Methods sampleMethod() {
        return SampleOperator.Methods.RANDOM;
    }

}
