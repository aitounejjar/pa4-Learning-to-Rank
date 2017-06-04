package cs276.pa4;

import cs276.pa3.BM25Scorer;
import cs276.pa3.SmallestWindowScorer;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

@SuppressWarnings("Duplicates")
public class Task3Learner extends PairwiseLearner {

    private static BM25Scorer bm25Scorer;
    private static SmallestWindowScorer smallestWindowScorer;

    // -----------------------------------------------------------------------------------------------------------------
    // enabling/disabling features
    // -----------------------------------------------------------------------------------------------------------------
    private static final boolean BM25_FEATURE_ENABLED            = true;
    private static final boolean SMALLEST_WINDOW_FEATURE_ENABLED = false;
    private static final boolean PAGE_RANK_FEATURE_ENABLED       = false;



    // -----------------------------------------------------------------------------------------------------------------
    // constructor(s)
    // -----------------------------------------------------------------------------------------------------------------

    public Task3Learner(boolean isLinearKernel) {
        super(isLinearKernel);
    }

    public Task3Learner(double C, double gamma, boolean isLinearKernel) {
        super(C, gamma, isLinearKernel);
    }


    // -----------------------------------------------------------------------------------------------------------------
    // overridden method(s)
    // -----------------------------------------------------------------------------------------------------------------

    @Override
    public Instances extractTrainFeatures(String train_data_file, String train_rel_file, Map<String, Double> idfs) {

        Feature feature = new Feature(idfs);

        Map<Query, List<Document>> trainData  = Util.loadTrainData_helper(train_data_file);

        Map<String, Map<String, Double>> relData = Util.loadRelData_helper(train_rel_file);

        /*
         * @TODO: Below is a piece of sample code to show
         * you the basic approach to construct a Instances
         * object, replace with your implementation.
         */

        Map<Query, Map<String, Document>> queryDict = buildQueryDict(trainData);

        // initialize the scorers from PA3
        bm25Scorer = new BM25Scorer(idfs, queryDict);
        smallestWindowScorer = new SmallestWindowScorer(idfs, queryDict);

        Instances standardized = new Instances("normalized", createAttributes_helper(), 0);
        Instances train_dataset = new Instances("train_data_set", createAttributes_helper(), 0);
        /* --------------------- Set last attribute as target --------------------- */
        train_dataset.setClassIndex(train_dataset.numAttributes() - 1);

        // loop to standardize
        Map<Query, Map<Document, Integer>> index_map = new HashMap<Query, Map<Document, Integer>>();
        for (Query q : trainData.keySet()) {
            Map<Document, Integer> docIndexes = new HashMap<Document, Integer>();
            for (Document d : trainData.get(q)) {
                double[] basic_features = feature.extractFeatureVector(d, q);
                double[] extra_features = toArray(getExtraFeatures(q, d));
                double[] features = augmentArray(basic_features, extra_features);
                Instance inst = new DenseInstance(1.0, features);
                standardized.add(inst);
                docIndexes.put(d, standardized.size());
            }
            index_map.put(q, docIndexes);
        }

        standardized = standardize(standardized);

        // compute the differences ...

        for (Query q : trainData.keySet()) {
            for (Document d1 : trainData.get(q)) {
                for (Document d2 : trainData.get(q)) {

                    // -- get the relevance :
                    double rel1 = relData.get(q.query).get(d1.url);
                    double rel2 = relData.get(q.query).get(d2.url);

                    if (d1.equals(d2) || rel1 == rel2) {
                        continue;
                    }

                    double[] d1_vector = standardized.get(index_map.get(q).get(d1)-1).toDoubleArray();
                    double[] d2_vector = standardized.get(index_map.get(q).get(d2)-1).toDoubleArray();

                    // -- compare relevances to decide whether d1 is better than d2,
                    //    if (d1>d2) then d1 is gonna be "positive", else d1 is negative.

                    String documentClass = (rel1 > rel2) ? "positive" : "negative";

                    double[] difference = compute_difference(d1_vector, d2_vector);
                    Instance instance = new DenseInstance(1.0, difference);
                    instance.insertAttributeAt(instance.numAttributes());
                    instance.setValue(train_dataset.attribute(instance.numAttributes() - 1), documentClass);
                    train_dataset.add(instance);

                }
            }
        }

        return train_dataset;
    }

    @Override
    public TestFeatures extractTestFeatures(String test_data_file, Map<String, Double> idfs) {

        Map<Query, List<Document>> trainData  = Util.loadTrainData_helper(test_data_file);

        Instances test_dataset = new Instances("test_dataset", createAttributes_helper(), 0);
        Map<Query, Map<Document, Integer>> index_map = new HashMap<Query, Map<Document, Integer>>();
        for (Query q : trainData.keySet()) {
            index_map.put(q, new HashMap<Document, Integer>());
            for (Document d1 : trainData.get(q)) {
                for (Document d2 : trainData.get(q)) {

                    if (d1.equals(d2)) {
                        continue;
                    }

                    test_dataset = standardize(test_dataset);

                    // tf-idf scores of the 5 fields
                    Feature feature = new Feature(idfs);
                    double[] tf_idf_scores1 = feature.extractFeatureVector(d1, q);
                    double[] tf_idf_scores2 = feature.extractFeatureVector(d2, q);

                    // compute the diff between the previous two vectors
                    double[] difference1 = compute_difference(tf_idf_scores2, tf_idf_scores1);

                    // we compute the -difference to balance classes
                    double[] difference2 = compute_difference(tf_idf_scores1, tf_idf_scores2);

                    // get extra features, i.e. bm25, smallestWindow, etc.
                    Pair<ArrayList<Double>, ArrayList<Double>> pair = getExtraFeatures(q, d1, d2);
                    ArrayList<Double> extra = pair.getFirst();
                    ArrayList<Double> extraBalancer = pair.getSecond();

                    double[] augmented_difference1 = augmentArray(difference1, toArray(extra));
                    double[] augmented_difference2 = augmentArray(difference2, toArray(extraBalancer));

                    // create the instance and add in the training set
                    test_dataset.add(new DenseInstance(1.0, augmented_difference1));
                    index_map.get(q).put(d1, test_dataset.size()-1);

                }

            }
        }

        Instances standardized = standardize(test_dataset);

        return new TestFeatures(standardized, index_map);
    }

    @Override
    public Map<Query, List<Document>> testing(TestFeatures tf, Classifier model) {
        Map<Query, List<Document>> rankedDocs = new HashMap<Query, List<Document>>();

        for (Query q : tf.index_map.keySet()) {

            Set<Document> set = tf.index_map.get(q).keySet();
            List<Document> docs = new ArrayList<Document>(set);
            Map<Document, Integer> map = tf.index_map.get(q);
            Collections.sort(docs, new Comparator<Document>() {
                @Override
                public int compare(Document d1, Document d2) {
                    //  - comparator should compare Documents
                    //  - the call to model.classifyInstance(...) should be done here
                    //  - the diff between the vectors of the two documents should be passed to SVM in the previous step

                    Integer index_of_d1 = map.get(d1);
                    Integer index_of_d2 = map.get(d2);

                    Instance instance1 = tf.features.get(index_of_d1);
                    Instance instance2 = tf.features.get(index_of_d2);

                    double vector1[] = instance1.toDoubleArray();
                    double vector2[] = instance2.toDoubleArray();

                    double[] difference = compute_difference(vector2, vector1);
                    Instance instance = new DenseInstance(1.0, difference);

                    Instances instances = new Instances("comparison_dataset", createAttributes_helper(), 0);
                    instances.setClassIndex(instances.numAttributes()-1);
                    instance.setDataset(instances);

                    Double guessedClass = 0.0;
                    try {
                        guessedClass = model.classifyInstance(instance);
                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }

                    return (guessedClass > 0) ? -1 : +1;
                }
            });

            rankedDocs.put(q, docs);
        }

        return rankedDocs;
    }

    // -----------------------------------------------------------------------------------------------------------------
    // private method(s) : helpers
    // -----------------------------------------------------------------------------------------------------------------

    /**
     * Creates and returns an array consisting of the passed base array, appended with the values passed in the
     * second argument.
     *
     * @param arr    base array
     * @param values array whose values are to be appended to the base array
     * @return       array of doubles
     */
    private double[] augmentArray(double[] arr, double ... values) {

        int size = arr.length + values.length;
        double[] augmented = new double[size];
        // copy the passed array to the new
        System.arraycopy(arr, 0, augmented, 0, arr.length);

        // append the extra values
        int startIndex = arr.length;
        for (int i=0; i<values.length; ++i) {
            augmented[startIndex+i] = values[i];
        }

        return augmented;
    }

    /**
     * @return list of attributes
     */
    private ArrayList<Attribute> createAttributes_helper() {

        ArrayList<String> list = new ArrayList<>();
        if (BM25_FEATURE_ENABLED)            list.add("bm25_w");
        if (SMALLEST_WINDOW_FEATURE_ENABLED) list.add("smallestWindow_w");
        if (PAGE_RANK_FEATURE_ENABLED)       list.add("pageRank_w");

        String[] array = new String[list.size()];
        return createAttributes(true, list.toArray(array) );
    }

    /**
     * builds a query dict that can be passed to public constructors of PA3 scorers
     * @param trainData training data
     * @return          map of the training data
     */
    private Map<Query,Map<String, Document>> buildQueryDict(Map<Query, List<Document>> trainData) {
        Map<Query, Map<String, Document>> queryDict = new HashMap<Query, Map<String, Document>>();

        for (Query q : trainData.keySet()) {
            int counter = 0;
            Map<String, Document> map = new HashMap<String, Document>();
            for (Document d : trainData.get(q)) {
                map.put(counter+"|"+d.url, d);
                counter++;
            }
            queryDict.put(q, map);
        }

        return queryDict;
    }

    // -----------------------------------------------------------------------------------------------------------------
    // private method(s): scorers, i.e. these methods compute scores of the different features
    // -----------------------------------------------------------------------------------------------------------------

    private Pair<ArrayList<Double>, ArrayList<Double>> getExtraFeatures(Query q, Document d1, Document d2) {

        ArrayList<Double> values = new ArrayList<Double>();
        ArrayList<Double> oppositeValues = new ArrayList<Double>();

        if (BM25_FEATURE_ENABLED) {
            // -- bm25 feature
            double score1 = getBM25Score(q, d1);
            double score2 = getBM25Score(q, d2);
            double diff = score2 - score1;
            values.add(diff);
            oppositeValues.add(-diff);
        }

        if (SMALLEST_WINDOW_FEATURE_ENABLED) {
            // -- smallest window feature
            double score1 = getSmallestWindowScore(q, d1);
            double score2 = getSmallestWindowScore(q, d2);
            double diff = score2 - score1;
            values.add(diff);
            oppositeValues.add(-diff);
        }

        if (PAGE_RANK_FEATURE_ENABLED) {
            // -- pagerank feature
            double delta_pagerank = d2.page_rank - d1.page_rank;
            values.add(delta_pagerank);
            oppositeValues.add(-delta_pagerank);
        }

        return new Pair<ArrayList<Double>, ArrayList<Double>>(values, oppositeValues);

    }

    private ArrayList<Double> getExtraFeatures(Query q, Document d) {
        ArrayList<Double> values = new ArrayList<Double>();

        if (BM25_FEATURE_ENABLED) {
            // -- bm25 feature
            double score = getBM25Score(q, d);
            values.add(score);
        }

        if (SMALLEST_WINDOW_FEATURE_ENABLED) {
            // -- smallest window feature
            double score = getSmallestWindowScore(q, d);
            values.add(score);
        }

        if (PAGE_RANK_FEATURE_ENABLED) {
            // -- pagerank feature
            values.add((double)d.page_rank);
        }

        return values;
    }

    /**
     * Returns the bm25 score of the (query,document) pair -- this uses PA3 code
     * @param q
     * @param d
     * @return score
     */
    private double getBM25Score(Query q, Document d) {
        return bm25Scorer.getSimScore(d, q);
    }

    /**
     * Returns the smallest window score of the (query,document) pair -- this uses PA3 code
     * @param q
     * @param d
     * @return score
     */
    private double getSmallestWindowScore(Query q, Document d) {
        return smallestWindowScorer.getSimScore(d, q);
    }

    private double getPdfValue(Document d) {
        return d.url.toLowerCase().endsWith(".pdf") ? 1.0 : 0.0;
    }

}
