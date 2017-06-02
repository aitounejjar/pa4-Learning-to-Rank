package cs276.pa4;

import weka.classifiers.Classifier;
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

    // private static data
    private static final String PDF_FEATURE_ID = "isPdf_w";
    private static final String BM25_FEATURE_ID = "BM25_w";

    // constructor(s)

    public Task3Learner(boolean isLinearKernel) {
        super(isLinearKernel);
    }

    public Task3Learner(double C, double gamma, boolean isLinearKernel) {
        super(C, gamma, isLinearKernel);
    }

    // overridden method(s)

    @Override
    public Instances extractTrainFeatures(String train_data_file, String train_rel_file, Map<String, Double> idfs) {

        Map<Query, List<Document>> trainData  = Util.loadTrainData_helper(train_data_file);

        Map<String, Map<String, Double>> relData = Util.loadRelData_helper(train_rel_file);

        /*
         * @TODO: Below is a piece of sample code to show
         * you the basic approach to construct a Instances
         * object, replace with your implementation.
         */

        Instances train_dataset = new Instances("train_data_set", createAttributes(true, BM25_FEATURE_ID), 0);

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
                    double[] difference = compute_difference(tf_idf_scores2, tf_idf_scores1);

                    // -- isPdf feature
                    double pdf1 = getPdfValue(d1);
                    double pdf2 = getPdfValue(d2);

                    double[] augmented_difference = augmentArray(difference, pdf2-pdf1);

                    // -- compare the relevance score to decide the : diff[5]

                    // -- get the relevance same way I did for task1:
                    double rel1 = relData.get(q.query).get(d1.url);
                    double rel2 = relData.get(q.query).get(d2.url);

                    // -- compare and decide whether d1 is better than d2,
                    //    if (d1>d2) then d1 is gonna be "positive", else d1 is negative.
                    String documentClass = "";
                    if (rel1 > rel2) {
                        documentClass = "positive";
                    } else if (rel1 < rel2) {
                        documentClass = "negative";
                    } else {
                        // do nothing
                        continue;
                    }

                    Instance instance = new DenseInstance(1.0, augmented_difference);
                    instance.insertAttributeAt(instance.numAttributes());
                    instance.setValue( train_dataset.attribute(instance.numAttributes() - 1), documentClass );
                    train_dataset.add(instance);

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

        Instances dataset = new Instances("test_dataset", createAttributes(true, BM25_FEATURE_ID), 0);
        Map<Query, Map<Document, Integer>> index_map = new HashMap<Query, Map<Document, Integer>>();
        for (Query q : trainData.keySet()) {
            index_map.put(q, new HashMap<Document, Integer>());
            for (Document d : trainData.get(q)) {


                // tf-idf scores of the 5 fields
                Feature feature = new Feature(idfs);
                double[] tf_idf_scores = feature.extractFeatureVector(d, q);
                double[] augmented_array = augmentArray(tf_idf_scores, getBM25Value());
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

                    Instances instances = new Instances("comparison_dataset", createAttributes(true, BM25_FEATURE_ID), 0);
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

    // private method(s)

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

    private double getPdfValue(Document d) {
        return d.url.toLowerCase().endsWith(".html") ? 1.0 : 1.0;
    }

    private double getBM25Value() {
        return 0.0;
    }
}
