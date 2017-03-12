package org.qcri.ml4all.abstraction.plan.wrappers;

import org.qcri.ml4all.abstraction.api.Update;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

import java.util.List;

/**
 * Created by zoi on 25/1/15.
 */
public class AssignWrapper<R> extends LogicalOperatorWrapperWithContext<ML4allContext, List<R>> { //TODO:check why this does not work because of the List<V> generic type

    Update<R,?> logOp;

    public AssignWrapper(Update logOp) {
        this.logOp = logOp;
    }

    @Override
    public ML4allContext apply(List<R> o) {
        ML4allContext newContext = context.clone();
        return this.logOp.assign(o, newContext);
    }

    @Override
    public void initialise() {

    }

//    @Override
//    public void open(ExecutionContext executionContext) {
//        context = executionContext.<ML4allContext>getBroadcast("context").iterator().next();
//
//    }
}
