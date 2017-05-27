package cs276.pa4;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.Instances;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Implements point-wise learner that can be used to implement logistic regression
 */
public class PointwiseLearner extends Learner {

    @Override
    public Instances extractTrainFeatures(String train_data_file, String train_rel_file, Map<String, Double> idfs) {

        Map<Query, List<Document>> trainData  = null;
        try {
            trainData = Util.loadTrainData(train_data_file);
        } catch (Exception e) {
            e.printStackTrace();
        }

        Map<String, Map<String, Double>> relData = null;
        try {
            relData = Util.loadRelData(train_rel_file);
        } catch (IOException e) {
            e.printStackTrace();
        }
    
        /*
         * @TODO: Below is a piece of sample code to show
         * you the basic approach to construct a Instances
         * object, replace with your implementation.
         */

        Instances dataset = null;
    
        /* --------------------- Build attributes list --------------------- */
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        attributes.add(new Attribute("url_w"));
        attributes.add(new Attribute("title_w"));
        attributes.add(new Attribute("body_w"));
        attributes.add(new Attribute("header_w"));
        attributes.add(new Attribute("anchor_w"));
        attributes.add(new Attribute("relevance_score"));
        dataset = new Instances("train_dataset", attributes, 0);
    
        /* --------------------- Add data --------------------- */

        // create a five-dimensional vector of tf-idf scores, for each query document pair
        for (Query q : trainData.keySet()) {
            String query = q.query;
            for (Document d : trainData.get(q)) {
                double rel = relData.get(query).get(d.url);

            }
        }

        /*double[] instance = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        Instance inst = new DenseInstance(1.0, instance);
        dataset.add(inst);*/
    
        /* --------------------- Set last attribute as target --------------------- */
        dataset.setClassIndex(dataset.numAttributes() - 1);

        return dataset;
    }

    @Override
    public Classifier training(Instances dataset) {
        /*
         * @TODO: Your code here
         */

        LinearRegression classifier = new LinearRegression();
        try {
            classifier.buildClassifier(dataset);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return classifier;
    }

    @Override
    public TestFeatures extractTestFeatures(String test_data_file,
                                            Map<String, Double> idfs) {
        /*
         * @TODO: Your code here
         * Create a TestFeatures object
         * Build attributes list, instantiate an Instances object with the attributes
         * Add data and populate the TestFeatures with the dataset and features
         */

        TestFeatures testFeatures = new TestFeatures();



        return testFeatures;
    }

    @Override
    public Map<Query, List<Document>> testing(TestFeatures tf,
                                              Classifier model) {
        /*
         * @TODO: Your code here
         */
        return null;
    }

}
