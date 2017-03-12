package org.qcri.ml4all.abstraction.plan.wrappers;

import org.qcri.ml4all.abstraction.api.Loop;

/**
 * Created by zoi on 1/2/15.
 */
public class LoopCheckWrapper<V> extends LogicalOperatorWrapperWithContext<Boolean, V> {

    Loop<V, Boolean> logOp;

    public LoopCheckWrapper(Loop logOp) {
        this.logOp = logOp;
    }

    @Override
    public Boolean apply(V o) {
        return this.logOp.terminate(o);
    }

    @Override
    public void initialise() { logOp.initialise(); }
}
