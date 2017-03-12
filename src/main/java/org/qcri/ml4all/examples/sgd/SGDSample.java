package org.qcri.ml4all.examples.sgd;

import org.qcri.ml4all.abstraction.api.Loop;
import org.qcri.ml4all.abstraction.api.Sample;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.basic.operators.SampleOperator;

public class SGDSample extends Sample {


    @Override
    public int sampleSize() {
        return 1;
    }

    @Override
    public SampleOperator.Methods sampleMethod() {
        return SampleOperator.Methods.ANY;
    }
}
