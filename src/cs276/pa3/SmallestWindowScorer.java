package cs276.pa3;

import com.sun.tools.javac.util.Assert;
import cs276.pa4.Document;
import cs276.pa4.Pair;
import cs276.pa4.Query;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A skeleton for implementing the Smallest Window scorer in Task 3.
 * Note: The class provided in the skeleton code extends BM25Scorer in Task 2. However, you don't necessarily
 * have to use Task 2. (You could also use Task 1, in which case, you'd probably like to extend CosineSimilarityScorer instead.)
 * Also, feel free to modify or add helpers inside this class.
 */
public class SmallestWindowScorer extends CosineSimilarityScorer {

    protected static final double BOOST_FACTOR = 2;
    protected static final double CUT_OFF = 9.0;


    public SmallestWindowScorer(Map<String, Double> idfs, Map<Query, Map<String, Document>> queryDict) {
        super(idfs);
    }

    /**
     * get smallest window of one document and query pair.
     *
     * @param d: document
     * @param q: query
     */
    protected Pair<Integer, Integer> getWindow(Document d, Query q) {
    /*
     * @//TODO : Your code here
     */
        List<Pair<Integer, Integer>> windows = new ArrayList<>();
        windows.add(new Pair(Integer.MAX_VALUE, 0));

        // compute smallest window for the body
        if (d.body_hits != null) {
            List<List<Integer>> positions = new ArrayList<>(d.body_hits.values());
            if (positions.size() == q.queryWords.size()) {
                // all query words appear in the body
                int[] window = computeSmallestWindow(positions);
                windows.add(new Pair<>(window[0], window[1]));
            }
        }

        // compute smallest window for url
        int[] window = computeSmallestWindow(d.url, q);
        windows.add(new Pair<>(window[0], window[1]));

        // compute smallest window for title
        window = computeSmallestWindow(d.title, q);
        windows.add(new Pair<>(window[0], window[1]));

        // compute smallest window for headers
        if (d.headers != null) {
            for (String header : d.headers) {
                window = computeSmallestWindow(header, q);
                windows.add(new Pair<>(window[0], window[1]));
            }
        }

        // compute smallest window for anchors
        if (d.anchors != null) {
            for (String header : d.anchors.keySet()) {
                window = computeSmallestWindow(header, q);
                windows.add(new Pair<>(window[0], window[1]));
            }
        }

        Collections.sort(windows, new Comparator<Pair<Integer, Integer>>() {
            @Override
            public int compare(Pair<Integer, Integer> o1, Pair<Integer, Integer> o2) {
                Integer w1 = o1.getFirst();
                Integer w2 = o2.getFirst();
                return w1.compareTo(w2);
            }
        });

        return windows.get(0);
    }


    /**
     * get boost score of one document and query pair.
     *
     * @param d: document
     * @param q: query
     */
    protected double getBoostScore(Document d, Query q) {

        /*
         * @//TODO : Your code here, calculate the boost score.
         *
         */

        // number of unique words in the query
        int querySize = new HashSet<>(q.queryWords).size();

        if (querySize == 1) {
            return 1.0;
        }

        Pair<Integer, Integer> pair = getWindow(d, q);
        int smallestWindow = pair.getFirst();

        double boostScore;

        if (smallestWindow == Integer.MAX_VALUE) {
            boostScore = 1.0;
        } else if (smallestWindow == querySize) {
            boostScore = BOOST_FACTOR * smallestWindow;
        } else {
            int delta = smallestWindow - querySize;
            boostScore = getBoostScore_helper(delta);

        }

        return boostScore;
    }

    protected double getBoostScore_helper(int x) {
        double d;
        if (x > CUT_OFF) {
            d = 1.0;
        } else {
            d = Math.abs(10.0 + ((1.0 - ((Math.pow(x,2)) / CUT_OFF)) * BOOST_FACTOR));
            d = Math.sqrt(d);
        }

        return d;
    }

    @Override
    public double getSimScore(Document d, Query q) {
        Map<String, Map<String, Double>> tfs = this.getDocTermFreqs(d, q);
        this.normalizeTFs(tfs, d, q);
        Map<String, Double> tfQuery = getQueryFreqs(q);
        double boost = getBoostScore(d, q);
        double rawScore = this.getNetScore(tfs, q, tfQuery, d);

        return ( (boost * rawScore) );
    }


    /**
     *
     * Returns a two-elements int array, where first element is the smallest window, and the second is
     * the number of times this smallest window was encountered
     *
     * @param string String in which to look for the smallest window
     * @param q      query
     * @return       int array
     */
    private int[] computeSmallestWindow(String string, Query q) {
        int smallestWindow = Integer.MAX_VALUE;

        // smallest window is infinity, if there is a single query word that doesn't appear in the string
        for (String w : q.queryWords) {
            if (!string.contains(w)) {
                return new int[] {smallestWindow, 0};
            }
        }

        // all query words appear in the passed strings
        String[] tokens = string.split("\\W+");

        Map<String, List<Integer>> positions = new HashMap<>();
        for (int i=0; i<tokens.length; ++i) {
            String token = tokens[i];
            if (!positions.containsKey(token)) {
                positions.put(token, new ArrayList<Integer>());
            }
            positions.get(token).add(i);
        }

        List<List<Integer>> lists = new ArrayList<>();
        for (String k : positions.keySet()) {
            lists.add(positions.get(k));
        }

        return computeSmallestWindow(lists);
    }

    private int[] computeSmallestWindow(List<List<Integer>> positions) {

        // pre-processing: put a mapping from the minimums to the list which they came from
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (List<Integer> list : positions) {
            Set<Integer> set = new HashSet<>(list);
            list = new ArrayList<>(set);
            if (set.size() != list.size()){
                Assert.check(set.size() == list.size(), "List of positions contained duplicated. List Size: "
                        + list.size() + ", Set Size: " + set.size() + "\n" + list.toString());
            }

            Collections.sort(list);
            int min = list.get(0);
            map.put(min, list);
        }

        int smallestWindow = computeSmallestWindow_helper(map.keySet());
        int smallestWindowCount = 1; // tracks how many times the smallest window was found

        while (true) {
            if (smallestWindow == positions.size() || allListsTraversed(map)) {
                break;
            }

            // find the minimum whose list can be increased
            List<Integer> mins = new ArrayList<>(map.keySet());
            Collections.sort(mins);
            for (int i=0; i<mins.size(); ++i) {
                // get the current minimum
                int currentMin = mins.get(i);
                // get the list where it came from
                List<Integer> list = map.get(currentMin);
                // get the currentMin's index in its list
                int index = list.indexOf(currentMin);
                if (index < list.size() - 1) {
                    int newMin = list.get(index+1);
                    map.remove(currentMin);
                    map.put(newMin, list);
                }
            }

            // by now, a new minimum has already been put to the map
            // we just need to compute the smallest window again

            int possibleSmallestWindow = computeSmallestWindow_helper(map.keySet());

            if (possibleSmallestWindow < smallestWindow) {
                smallestWindow = possibleSmallestWindow;
                smallestWindowCount = 1; // reset the count because a new smallest window was found
            } else if (possibleSmallestWindow == smallestWindow) {
                smallestWindowCount++;
            }

        }

        return new int[] {smallestWindow, smallestWindowCount};

    }

    private int computeSmallestWindow_helper(Set<Integer> set) {
        int size = set.size();
        List<Integer> list = new ArrayList<>(set);
        Collections.sort(list);
        int range = list.get(size-1) - list.get(0) + 1;
        return range;
    }

    private boolean allListsTraversed(Map<Integer, List<Integer>> map) {

        boolean result = true;

        for (int minimum : map.keySet()) {
            List<Integer> list = map.get(minimum);
            int index = list.indexOf(minimum);
            if (index < list.size() - 1) {
                result = false;
                break;
            }
        }

        return result;
    }

}
