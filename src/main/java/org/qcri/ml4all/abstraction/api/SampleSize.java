package org.qcri.ml4all.abstraction.api;

/**
 * Created by zoi on 22/1/15.
 */
public abstract class SampleSize extends LogicalOperator {

    public abstract long getSampleSize (int currentIteration);

}
