package cs276.pa4;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

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

        /*
         * @TODO: Below is a piece of sample code to show
         * you the basic approach to construct a Instances
         * object, replace with your implementation.
         */

        Instances train_dataset = new Instances("train_data_set", createAttributes(), 0);

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

                    Instance instance = new DenseInstance(1.0, difference);
                    instance.insertAttributeAt(instance.numAttributes());
                    instance.setValue(train_dataset.attribute(instance.numAttributes() - 1), documentClass);
                    train_dataset.add(instance);

                }
            }
        }

        /* --------------------- Set last attribute as target --------------------- */
        train_dataset.setClassIndex(train_dataset.numAttributes() - 1);

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

        Instances dataset = new Instances("test_dataset", createAttributes(), 0);
        Map<Query, Map<Document, Integer>> index_map = new HashMap<Query, Map<Document, Integer>>();
        for (Query q : trainData.keySet()) {
            index_map.put(q, new HashMap<Document, Integer>());
            for (Document d : trainData.get(q)) {

                double vector[] = new double[6];
                vector[5]=0;

                // tf-idf scores of the 5 fields
                Feature feature = new Feature(idfs);
                double[] tf_idf_scores = feature.extractFeatureVector(d, q);
                System.arraycopy(tf_idf_scores, 0, vector, 0, 5);

                dataset.add(new DenseInstance(1.0, vector));

                index_map.get(q).put(d, dataset.size()-1);
            }
        }

        Instances standardized = standardize(dataset);

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

                    Instances instances = new Instances("comparison_dataset", createAttributes(), 0);
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

    private ArrayList<Attribute> createAttributes() {

        // add attributes representing each of the 5 sections
        String[] arr = new String[]{"url_w", "title_w", "body_w", "header_w", "anchor_w"};
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i=0; i<arr.length; ++i) {
            attributes.add(new Attribute(arr[i]));
        }

        // add document classes
        ArrayList<String> classes = new ArrayList<>();
        classes.add("positive");
        classes.add("negative");
        attributes.add(new Attribute("document_classes", classes));

        return attributes;
    }

    private Instances standardize(Instances dataset) {
        Standardize filter = new Standardize();
        // Normalize filter = new Normalize(); filter.setScale(2.0); filter.setTranslation(-1.0); // scale values to [-1, 1]
        Instances standardizedInstance = null;
        try {
            filter.setInputFormat(dataset);
            standardizedInstance = Filter.useFilter(dataset, filter);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return standardizedInstance;
    }

    private double[] compute_difference(double[] a, double[] b) {
        double[] difference = new double[a.length];
        for (int i=0; i<a.length; ++i) {
            difference[i] = b[i] - a[i];
        }
        return difference;
    }

}
