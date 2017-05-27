package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Query implements Comparable<Query> {
    String query;
    List<String> queryWords;

    public Query(String query) {
        this.query = new String(query);
        String[] wordsArray = query.toLowerCase().split(" ");
        queryWords = new ArrayList<String>(Arrays.asList(wordsArray));
    }

    @Override
    public int compareTo(Query arg0) {
        return this.query.compareTo(arg0.query);
    }

    @Override
    public String toString() {
        return query;
    }
}