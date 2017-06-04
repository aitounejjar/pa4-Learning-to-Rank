package cs276.pa4;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * An abstract class that is extended by PairwiseLearner and PointWiseLearner
 */
public abstract class Learner {
    public static boolean isLinearKernel = false;

    /* Construct training features matrix */
    public abstract Instances extractTrainFeatures(String train_data_file, String train_rel_file, Map<String, Double> idfs);

    /* Train the model */
    public abstract Classifier training(Instances dataset);

    /* Construct testing features matrix */
    public abstract TestFeatures extractTestFeatures(String test_data_file, Map<String, Double> idfs);

    /* Test the model, return ranked queries */
    public abstract Map<Query, List<Document>> testing(TestFeatures tf, Classifier model);

    public Instances standardize(Instances dataset) {
        Standardize filter = new Standardize();
        //Normalize filter = new Normalize(); filter.setScale(2.0); filter.setTranslation(-1.0); // scale values to [-1, 1]
        Instances standardizedInstance = null;
        try {
            filter.setInputFormat(dataset);
            standardizedInstance = Filter.useFilter(dataset, filter);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return standardizedInstance;
    }

    public double[] compute_difference(double[] a, double[] b) {
        double[] difference = new double[a.length];
        for (int i=0; i<a.length; ++i) {
            difference[i] = b[i] - a[i];
        }
        return difference;
    }

    public ArrayList<Attribute> createAttributes(boolean includeDocumentClasses, String ... extraAttributes) {

        // add attributes representing each of the 5 sections
        String[] arr = new String[]{"url_w", "title_w", "body_w", "header_w", "anchor_w"};
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i=0; i<arr.length; ++i) {
            attributes.add(new Attribute(arr[i]));
        }

        // add any extra attributes
        for (String extra : extraAttributes) {
            attributes.add(new Attribute(extra));
        }

        if (includeDocumentClasses) {
            // add document classes
            ArrayList<String> classes = new ArrayList<>();
            classes.add("positive");
            classes.add("negative");
            attributes.add(new Attribute("document_classes", classes));
        }

        return attributes;
    }

    /**
     * @param list list to be converted to an array of doubles
     * @return array of doubles
     */
    public double[] toArray(ArrayList<Double> list) {

        if (list == null || list.isEmpty()) {
            return new double[0];
        }

        double[] d = new double[list.size()];

        for (int i=0; i<list.size(); ++i) {
            d[i] = list.get(i);
        }

        return d;
    }

}
