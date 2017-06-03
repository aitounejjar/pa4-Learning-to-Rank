package cs276.pa3;

import cs276.pa4.Document;
import cs276.pa4.Query;
import cs276.pa4.Util;

import java.util.HashMap;
import java.util.Map;

/**
 * An abstract class for a scorer.
 * Needs to be extended by each specific implementation of scorers.
 */
public abstract class AScorer {

    // Map: term -> idf
    Map<String, Double> idfs;

    // Various types of term frequencies that you will need
    String[] TFTYPES = {"url", "title", "body", "header", "anchor"};

    /**
     * Construct an abstract scorer with a map of idfs.
     *
     * @param idfs the map of idf scores
     */
    public AScorer(Map<String, Double> idfs) {
        this.idfs = idfs;
    }

    /**
     * You can implement your own function to whatever you want for debug string
     * The following is just an example to include page information in the debug string
     * The string will be forced to be 1-line and truncated to only include the first 200 characters
     */
    public String getDebugStr(Document d, Query q) {
        return "Pagerank: " + Integer.toString(d.page_rank);
    }

    /**
     * Score each document for each query.
     *
     * @param d the Document
     * @param q the Query
     */
    public abstract double getSimScore(Document d, Query q);

    /**
     * Get frequencies for a query.
     *
     * @param q the query to compute frequencies for
     */
    public Map<String, Double> getQueryFreqs(Query q) {

        // queryWord -> term frequency
        Map<String, Double> tfQuery = new HashMap<>();

        /*
         * TODO : Y o u r c o d e h e r e
         * Compute the raw term (and/or sublinearly scaled) frequencies
         * Additionally weight each of the terms using the idf value
         * of the term in the query (we use the PA1 corpus to determine
         * how many documents contain the query terms which is stored
         * in this.idfs).
         */

        for (String word : q.queryWords) {
            double count = 1;
            word = word.toLowerCase();
            if (tfQuery.containsKey(word)) {
                count += tfQuery.get(word);
            }
            tfQuery.put(word, count);
        }

        return tfQuery;
    }

    /**
     * Accumulate the various kinds of term frequencies
     * for the fields (url, title, body, header, and anchor).
     * You can override this if you'd like, but it's likely
     * that your concrete classes will share this implementation.
     *
     * @param d the Document
     * @param q the Query
     */
    public Map<String, Map<String, Double>> getDocTermFreqs(Document d, Query q) {
        return Util.getDocTermFreqs(d, q);
    }

    public int countOccurrences(String pattern, String string) {
        int count = 0;
        while (string.contains(pattern)) {
            ++count;
            string = string.replaceFirst(pattern, "");
        }

        return count;
    }

}
