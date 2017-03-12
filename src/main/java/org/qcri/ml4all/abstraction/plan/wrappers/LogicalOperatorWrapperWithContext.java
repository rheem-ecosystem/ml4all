package org.qcri.ml4all.abstraction.plan.wrappers;

import org.qcri.ml4all.abstraction.plan.context.ML4allContext;
import org.qcri.rheem.core.function.ExecutionContext;
import org.qcri.rheem.core.function.FunctionDescriptor;

/**
 * Created by zoi on 2/21/16.
 */
public abstract class LogicalOperatorWrapperWithContext<R, V> implements FunctionDescriptor.ExtendedSerializableFunction<V, R> {

    ML4allContext context;
    int currentIteration;
    private boolean first = true;

    @Override
    public void open(ExecutionContext executionContext) {
        currentIteration = executionContext.getCurrentIteration();
        context = executionContext.<ML4allContext>getBroadcast("context").iterator().next();
        if (this instanceof ComputePerPartitionWrapper) {
            if (first) {
                initialise();
                first = false;
            }
            else {
                first = true;
            }
        }
    }


    public abstract void initialise();

    public void finalise() { }
}
