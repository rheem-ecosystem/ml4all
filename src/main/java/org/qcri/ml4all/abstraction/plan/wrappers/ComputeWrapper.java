package org.qcri.ml4all.abstraction.plan.wrappers;

import org.qcri.ml4all.abstraction.api.Compute;

/**
 * Created by zoi on 25/1/15.
 */
public class ComputeWrapper<R, V> extends LogicalOperatorWrapperWithContext<R, V> {

    Compute<R, V> logOp;

    public ComputeWrapper(Compute logOp) {
        this.logOp = logOp;
    }

    @Override
    public R apply(V o) {
        return this.logOp.process(o, context);
    }

    @Override
    public void initialise() { logOp.initialise(); }

}
