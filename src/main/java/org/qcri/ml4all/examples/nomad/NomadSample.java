package org.qcri.ml4all.examples.nomad;

import org.qcri.ml4all.abstraction.api.Sample;
import org.qcri.rheem.basic.operators.SampleOperator;

public class NomadSample extends Sample {


    @Override
    public int sampleSize() {
        return 1;
    }

    @Override
    public SampleOperator.Methods sampleMethod() {
        return SampleOperator.Methods.ANY;
    }
}
