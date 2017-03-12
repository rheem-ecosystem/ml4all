package org.qcri.ml4all.abstraction.plan.wrappers;

import org.qcri.ml4all.abstraction.api.Transform;

/**
 * Created by zoi on 25/1/15.
 */
public class TransformWrapper<R, V> extends LogicalOperatorWrapper<R, V> {

    Transform<R, V> logOp;

    public TransformWrapper(Transform logOp) {
        this.logOp = logOp;
    }

    @Override
    public R apply(V o) {
        return this.logOp.transform(o);
    }
}
