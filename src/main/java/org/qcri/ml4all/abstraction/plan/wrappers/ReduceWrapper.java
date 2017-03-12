package org.qcri.ml4all.abstraction.plan.wrappers;

import org.qcri.rheem.basic.data.Tuple2;
import org.qcri.rheem.core.function.FunctionDescriptor;

import java.util.ArrayList;

/**
 * Created by zoi on 4/4/16.
 */
public class ReduceWrapper<T> implements FunctionDescriptor.SerializableBinaryOperator<T> {

    @Override
    public Object apply(Object o, Object o2) {
        ArrayList<Tuple2> a = (ArrayList<Tuple2>) o;
        ArrayList<Tuple2> b = (ArrayList<Tuple2>) o2;

        if (a == null)
            return b;
        else if (b == null)
            return a;
        else
            a.addAll(b);
        return a;
    }
}
