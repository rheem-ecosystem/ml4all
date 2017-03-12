package org.qcri.ml4all.abstraction.plan.wrappers;

import org.qcri.ml4all.abstraction.api.Loop;

/**
 * Created by zoi on 1/2/15.
 */
public class LoopConvergenceWrapper<R, V> extends LogicalOperatorWrapperWithContext<R, V> {

    Loop<R, V> logOp;

    public LoopConvergenceWrapper(Loop logOp) {
        this.logOp = logOp;
    }

    @Override
    public R apply(V o) {
        return this.logOp.prepareConvergenceDataset(o, context);
    }

    @Override
    public void initialise() { logOp.initialise(); }

}
