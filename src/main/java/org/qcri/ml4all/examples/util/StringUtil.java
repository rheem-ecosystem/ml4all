package org.qcri.ml4all.examples.util;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by zoi on 3/30/15.
 */
public class StringUtil {

    /**
     * Split a string with an unknown number of tokens with a given separator.
     *
     * @param string    the string to split
     * @param separator the separator
     * @return list of tokens obtained by splitting
     */
    public static List<String> split(String string, char separator) {
        final List<String> list = new ArrayList<>();
        int index = 0;
        int start = 0;
        final int len = string.length();
        while (index < len) {
            if (string.charAt(index) == separator) {
                list.add(string.substring(start, index));
                start = ++index;
            }
            else
                ++index;
        }
        if (start != index)
            list.add(string.substring(start, index));

        return list;
    }

    public static String[] split(String string, char separator, int limit) {
        final String[] result = new String[limit];
        int index = 0;
        int start = 0;
        final int len = string.length();
        int pos = 0;
        while (index < len) {
            if (string.charAt(index) == separator) {
                result[pos] = string.substring(start, index);
                ++pos;
                if (pos == limit) {
                    return result;
                }
                start = ++index;
            } else {
                ++index;
            }
        }
        if (start != index) {
            result[pos] = string.substring(start, index);
        }
        return result;
    }


}
