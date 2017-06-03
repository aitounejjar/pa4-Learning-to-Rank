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
    private static final boolean SMALLEST_WINDOW_FEATURE_ENABLED = true;
    private static final boolean PAGE_RANK_FEATURE_ENABLED       = true;

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

        Instances train_dataset = new Instances("train_data_set", createAttributes_helper(), 0);

        // compute the differences ...
        Map<Query, Map<Document, Integer>> index_map = new HashMap<Query, Map<Document, Integer>>();
        for (Query q : trainData.keySet()) {
            for (Document d1 : trainData.get(q)) {
                for (Document d2 : trainData.get(q)) {

                    if (d1.equals(d2)) {
                        continue;
                    }

                    // tf-idf scores of the 5 fields
                    Feature feature = new Feature(idfs);
                    double[] tf_idf_scores1 = feature.extractFeatureVector(d1, q);
                    double[] tf_idf_scores2 = feature.extractFeatureVector(d2, q);

                    // -- compute the diff between the previous two vectors
                    double[] difference1 = compute_difference(tf_idf_scores2, tf_idf_scores1);

                    // we compute the -difference to balance classes
                    double[] difference2 = compute_difference(tf_idf_scores1, tf_idf_scores2);

                    // -- isPdf feature
                    //double pdf1 = getPdfValue(d1);
                    //double pdf2 = getPdfValue(d2);

                    ArrayList<Double> extra = new ArrayList<>();
                    ArrayList<Double> extraBalancer = new ArrayList<>();

                    if (BM25_FEATURE_ENABLED) {
                        // -- bm25 feature
                        double score1 = getBM25Score(q, d1);
                        double score2 = getBM25Score(q, d2);
                        double diff = score2 - score1;
                        extra.add(diff);
                        extraBalancer.add(-diff);
                    }

                    if (SMALLEST_WINDOW_FEATURE_ENABLED) {
                        double score1 = getSmallestWindowScore(q, d1);
                        double score2 = getSmallestWindowScore(q, d2);
                        double diff = score2 - score1;
                        extra.add(diff);
                        extraBalancer.add(-diff);
                    }

                    if (PAGE_RANK_FEATURE_ENABLED) {
                        // -- pagerank feature
                        double delta_pagerank = d2.page_rank - d1.page_rank;
                        extra.add(delta_pagerank);
                        extraBalancer.add(-delta_pagerank);
                    }


                    double[] augmented_difference1 = augmentArray(difference1, toArray(extra));
                    double[] augmented_difference2 = augmentArray(difference2, toArray(extraBalancer));

                    // -- compare the relevance score to decide the : diff[5]

                    // -- get the relevance same way I did for task1:
                    double rel1 = relData.get(q.query).get(d1.url);
                    double rel2 = relData.get(q.query).get(d2.url);

                    // -- compare and decide whether d1 is better than d2,
                    //    if (d1>d2) then d1 is gonna be "positive", else d1 is negative.
                    String documentClass = "";
                    String balancingClass = "";
                    if (rel1 > rel2) {
                        documentClass = "positive";
                        balancingClass = "negative";
                    } else if (rel1 < rel2) {
                        documentClass = "negative";
                        balancingClass = "positive";
                    } else {
                        // do nothing
                        continue;
                    }

                    // create the instance and add in the training set
                    Instance instance1 = new DenseInstance(1.0, augmented_difference1);
                    instance1.insertAttributeAt(instance1.numAttributes());
                    instance1.setValue( train_dataset.attribute(instance1.numAttributes() - 1), documentClass );
                    train_dataset.add(instance1);

                    // balancing
                    Instance instance2 = new DenseInstance(1.0, augmented_difference2);
                    instance2.insertAttributeAt(instance2.numAttributes());
                    instance2.setValue( train_dataset.attribute(instance2.numAttributes() - 1), balancingClass );
                    train_dataset.add(instance2);

                }
            }
        }

        /* --------------------- Set last attribute as target --------------------- */
        train_dataset.setClassIndex(train_dataset.numAttributes() - 1);

        return train_dataset;
    }

    @Override
    public TestFeatures extractTestFeatures(String test_data_file, Map<String, Double> idfs) {
        Map<Query, List<Document>> trainData  = Util.loadTrainData_helper(test_data_file);

        Instances dataset = new Instances("test_dataset", createAttributes_helper(), 0);
        Map<Query, Map<Document, Integer>> index_map = new HashMap<Query, Map<Document, Integer>>();
        for (Query q : trainData.keySet()) {
            index_map.put(q, new HashMap<Document, Integer>());
            for (Document d : trainData.get(q)) {

                // tf-idf scores of the 5 fields
                Feature feature = new Feature(idfs);
                double[] tf_idf_scores = feature.extractFeatureVector(d, q);

                ArrayList<Double> extra = new ArrayList<>();

                if (BM25_FEATURE_ENABLED) {
                    extra.add(getBM25Score(q, d));
                }

                if (SMALLEST_WINDOW_FEATURE_ENABLED) {
                    extra.add(getSmallestWindowScore(q, d));
                }

                if (PAGE_RANK_FEATURE_ENABLED) {
                    extra.add((double)d.page_rank);
                }

                double[] augmented_array = augmentArray(tf_idf_scores, toArray(extra));
                dataset.add(new DenseInstance(1.0, augmented_array));
                index_map.get(q).put(d, dataset.size()-1);
            }
        }

        Instances standardized = standardize(dataset);

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

                    return (guessedClass > 0) ? +1 : -1;
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
        for (int i=startIndex; i<values.length; ++i) {
            augmented[i] = values[i];
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
     * @param list list to be converted to an array of doubles
     * @return array of doubles
     */
    private double[] toArray(ArrayList<Double> list) {

        if (list == null || list.isEmpty()) {
            return new double[0];
        }

        double[] d = new double[list.size()];

        for (int i=0; i<list.size(); ++i) {
            d[i] = list.get(i);
        }

        return d;
    }

    /**
     * builds a query dict that can be passed to public constructors of PA3 scorers
     * @param trainData training data
     * @return          map of the training data
     */
    private Map<Query,Map<String, Document>> buildQueryDict(Map<Query, List<Document>> trainData) {
        Map<Query, Map<String, Document>> queryDict = new HashMap<Query, Map<String, Document>>();

        for (Query q : trainData.keySet()) {

            Map<String, Document> map = new HashMap<String, Document>();
            for (Document d : trainData.get(q)) {
                map.put(d.url, d);
            }
            queryDict.put(q, map);
        }

        return queryDict;
    }

    // -----------------------------------------------------------------------------------------------------------------
    // private method(s): scorers, i.e. these methods compute scores of the different features
    // -----------------------------------------------------------------------------------------------------------------

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
