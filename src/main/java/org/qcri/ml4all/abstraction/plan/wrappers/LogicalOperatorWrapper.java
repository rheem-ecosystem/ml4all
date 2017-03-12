package org.qcri.ml4all.abstraction.plan.wrappers;

import org.qcri.rheem.core.function.FunctionDescriptor;

/**
 * Created by zoi on 2/21/16.
 */
public abstract class LogicalOperatorWrapper<R, V> implements FunctionDescriptor.SerializableFunction<V, R> {
    public void initialise() { }
    public void finalise() { }
}
