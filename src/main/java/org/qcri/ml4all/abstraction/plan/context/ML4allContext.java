package org.qcri.ml4all.abstraction.plan.context;

import gnu.trove.map.hash.THashMap;

import java.io.Serializable;

/**
 * For keeping global variables to access in the logical operators.
 */
public class ML4allContext implements Serializable {

    private THashMap<String, Object> map = new THashMap(32);

    public ML4allContext put(String key, Object value) {
        this.map.put(key, value);
        return this;
    }

    public Object getByKey(String key) {
        return this.map.get(key);
    }

    @Override
    public ML4allContext clone() {
        ML4allContext newContext = new ML4allContext();
        this.map.forEach((k,v) -> newContext.put(k, v));
        return newContext;
    }

    public String toString() {
        return map.toString();
    }

}
