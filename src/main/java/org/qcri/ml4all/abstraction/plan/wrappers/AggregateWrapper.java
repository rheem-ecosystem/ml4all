package org.qcri.ml4all.abstraction.plan.wrappers;

import org.qcri.ml4all.abstraction.api.Compute;
import org.qcri.rheem.core.function.FunctionDescriptor;

/**
 * Created by zoi on 4/4/16.
 */

public class AggregateWrapper<R> implements FunctionDescriptor.SerializableBinaryOperator<R> {
    Compute<R,?> logOp;

    public AggregateWrapper(Compute logOp) {
        this.logOp = logOp;
    }

    @Override
    public R apply(R o, R o2) {
        return this.logOp.aggregate(o, o2);
    }
}