package org.qcri.ml4all.abstraction.plan.wrappers;


import org.qcri.ml4all.abstraction.api.UpdateLocal;

/**
 * Created by zoi on 1/2/15.
 */
public class UpdateLocalWrapper<R, V> extends LogicalOperatorWrapperWithContext<R, V> {

    UpdateLocal<R, V> logOp;

    public UpdateLocalWrapper(UpdateLocal logOp) {
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
