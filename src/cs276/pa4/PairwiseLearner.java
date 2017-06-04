package cs276.pa4;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Implements Pairwise learner that can be used to train SVM
 */
public class PairwiseLearner extends Learner {
    private LibSVM model;

    public PairwiseLearner(boolean isLinearKernel) {
        try {
            model = new LibSVM();
        } catch (Exception e) {
            e.printStackTrace();
        }

        if (isLinearKernel) {
            model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
        }
    }

    public PairwiseLearner(double C, double gamma, boolean isLinearKernel) {
        try {
            model = new LibSVM();
        } catch (Exception e) {
            e.printStackTrace();
        }

        model.setCost(C);
        model.setGamma(gamma); // only matter for RBF kernel
        if (isLinearKernel) {
            model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
        }
    }

    @Override
    public Instances extractTrainFeatures(String train_data_file, String train_rel_file, Map<String, Double> idfs) {
        /*
         * @TODO: Your code here:
         * Get signal file
         * Construct output dataset of type Instances
         * Add new attribute to store relevance in the train dataset
         * Populate data
         */


        Map<Query, List<Document>> trainData  = Util.loadTrainData_helper(train_data_file);

        Map<String, Map<String, Double>> relData = Util.loadRelData_helper(train_rel_file);

        Feature feature = new Feature(idfs);

        Instances standardized = new Instances("normalized", createAttributes(true), 0);
        Instances train_dataset = new Instances("train_data_set", createAttributes(true), 0);
        /* --------------------- Set last attribute as target --------------------- */
        train_dataset.setClassIndex(train_dataset.numAttributes() - 1);

        // loop to standardize
        Map<Query, Map<Document, Integer>> index_map = new HashMap<Query, Map<Document, Integer>>();
        for (Query q : trainData.keySet()) {
            Map<Document, Integer> docIndexes = new HashMap<Document, Integer>();
            for (Document d : trainData.get(q)) {
                double[] features = feature.extractFeatureVector(d, q);
                Instance inst = new DenseInstance(1.0, features);
                standardized.add(inst);
                docIndexes.put(d, standardized.size());
            }
            index_map.put(q, docIndexes);
        }

        standardized = standardize(standardized);

        /*
         * @TODO: Below is a piece of sample code to show
         * you the basic approach to construct a Instances
         * object, replace with your implementation.
         */


        for (Query q : trainData.keySet()) {
            for (Document d1 : trainData.get(q)) {
                for (Document d2 : trainData.get(q)) {

                    // get the relevance
                    double rel1 = relData.get(q.query).get(d1.url);
                    double rel2 = relData.get(q.query).get(d2.url);

                    if (d1.equals(d2) || rel1 == rel2) {
                        continue;
                    }

                    // tf-idf scores of the 5 fields
                    double[] tf_idf_scores1 = standardized.get(index_map.get(q).get(d1)-1).toDoubleArray();
                    double[] tf_idf_scores2 = standardized.get(index_map.get(q).get(d2)-1).toDoubleArray();

                    // -- compute the diff between the previous two vectors
                    double[] difference = compute_difference(tf_idf_scores1, tf_idf_scores2);

                    // -- compare the relevance score to decide the : diff[5]

                    // -- compare and decide whether d1 is better than d2,
                    //    if (d1>d2) then d1 is gonna be "positive", else d1 is negative.
                    String documentClass = (rel1 > rel2) ? "positive" : "negative";

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
    public Classifier training(Instances dataset) {
        /*
         * @TODO: Your code here
         * Build classifier
         */
        try {
            model.buildClassifier(dataset);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return model;
    }

    @Override
    public TestFeatures extractTestFeatures(String test_data_file, Map<String, Double> idfs) {

        /*
         * @TODO: Your code here
         * Use this to build the test features that will be used for testing
         */

        Map<Query, List<Document>> trainData  = Util.loadTrainData_helper(test_data_file);

        Instances test_dataset = new Instances("test_dataset", createAttributes(true), 0);

        Map<Query, Map<Document, Integer>> index_map = new HashMap<Query, Map<Document, Integer>>();
        for (Query q : trainData.keySet()) {
            index_map.put(q, new HashMap<Document, Integer>());
            for (Document d1 : trainData.get(q)) {
                for (Document d2 : trainData.get(q)) {

                    if (d1.equals(d2)) {
                        continue;
                    }

                    // tf-idf scores of the 5 fields
                    Feature feature = new Feature(idfs);
                    double[] tf_idf_scores1 = feature.extractFeatureVector(d1, q);
                    double[] tf_idf_scores2 = feature.extractFeatureVector(d2, q);

                    // compute the diff between the previous two vectors
                    double[] difference1 = compute_difference(tf_idf_scores2, tf_idf_scores1);

                    // create the instance and add in the training set
                    test_dataset.add(new DenseInstance(1.0, difference1));
                    index_map.get(q).put(d1, test_dataset.size()-1);

                }

            }
        }

        Instances standardized = standardize(test_dataset);

        return new TestFeatures(standardized, index_map);
    }

    @Override
    // this is where the comparison happens
    public Map<Query, List<Document>> testing(TestFeatures tf, Classifier model) {
        /*
         * @TODO: Your code here
         */

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

                    Instances instances = new Instances("comparison_dataset", createAttributes(true), 0);
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

}
