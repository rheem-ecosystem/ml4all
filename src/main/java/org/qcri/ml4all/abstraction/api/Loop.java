package org.qcri.ml4all.abstraction.api;

import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

/**
 * Created by zoi on 22/1/15.
 */
public abstract class Loop<R, V> extends LogicalOperator {

    /* prepare the convergence dataset that will be used for the termination predicate
    * eg., the difference of the L2-norm of the new weights @input and the old weights which are in the context
    */
    public abstract R prepareConvergenceDataset(V input, ML4allContext context);

    /* given the output of the convergence dataset decide if you want to continue or not */
    public abstract boolean terminate(R input);

}
