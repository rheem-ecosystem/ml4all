package org.qcri.ml4all.abstraction.plan.wrappers;


import org.qcri.ml4all.abstraction.api.Update;

/**
 * Created by zoi on 1/2/15.
 */
public class UpdateWrapper<R, V> extends LogicalOperatorWrapperWithContext<R, V> {

    Update<R, V> logOp;

    public UpdateWrapper(Update logOp) {
        this.logOp = logOp;
    }

    @Override
    public R apply(V o) {
        return this.logOp.process(o, context);
    }

    @Override
    public void initialise() {

    }
}
