package org.qcri.ml4all.abstraction.api;

import org.qcri.rheem.basic.operators.SampleOperator;

/**
 * Created by zoi on 22/1/15.
 */
public abstract class Sample extends LogicalOperator {

    /* specify sample size */
    public abstract int sampleSize();

    /* specify which Sample method to use */
    public abstract SampleOperator.Methods sampleMethod();

}