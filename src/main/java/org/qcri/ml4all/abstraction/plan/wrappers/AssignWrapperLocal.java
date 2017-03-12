package org.qcri.ml4all.abstraction.plan.wrappers;

import org.qcri.ml4all.abstraction.api.UpdateLocal;
import org.qcri.ml4all.abstraction.plan.context.ML4allContext;

/**
 * Created by zoi on 25/1/15.
 */
public class AssignWrapperLocal<V> extends LogicalOperatorWrapperWithContext<ML4allContext, V> {

    UpdateLocal<V,?> logOp;

    public AssignWrapperLocal(UpdateLocal logOp) {
        this.logOp = logOp;
    }

    @Override
    public ML4allContext apply(V o) {
        ML4allContext newContext = context.clone();
        return this.logOp.assign(o, newContext);
    }

    @Override
    public void initialise() {

    }
}
