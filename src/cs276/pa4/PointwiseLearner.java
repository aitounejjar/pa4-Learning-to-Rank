package cs276.pa4;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
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

        dataset = new Instances("train_dataset", createAttributes(false, "relevance_score"), 0);
    
        /* --------------------- Add data --------------------- */

        // create a five-dimensional vector of tf-idf scores, for each query document pair
        for (Query q : trainData.keySet()) {
            String query = q.query;
            for (Document d : trainData.get(q)) {

                double vector[] = new double[6];

                // relevance
                double relevance = relData.get(query).get(d.url);
                vector[5] = relevance;

                // tf-idf scores of the 5 fields
                Feature feature = new Feature(idfs);
                double[] tf_idf_scores = feature.extractFeatureVector(d, q);
                System.arraycopy(tf_idf_scores, 0, vector, 0, 5);

                dataset.add(new DenseInstance(1.0, vector));

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

        Map<Query, List<Document>> trainData  = null;
        try {
            trainData = Util.loadTrainData(test_data_file);
        } catch (Exception e) {
            e.printStackTrace();
        }

        Instances dataset = null;
        Map<Query, Map<Document, Integer>> index_map = new HashMap<Query, Map<Document, Integer>>();

        dataset = new Instances("train_dataset", createAttributes(false, "relevance_score"), 0);

        /* --------------------- Add data --------------------- */
        // create a five-dimensional vector of tf-idf scores, for each query document pair
        for (Query q : trainData.keySet()) {
            String query = q.query;
            for (Document d : trainData.get(q)) {

                double vector[] = new double[6];
                int relevance = -1;
                vector[5] = relevance;

                // tf-idf scores of the 5 fields
                Feature feature = new Feature(idfs);
                double[] tf_idf_scores = feature.extractFeatureVector(d, q);
                System.arraycopy(tf_idf_scores, 0, vector, 0, 5);

                dataset.add(new DenseInstance(1.0, vector));

                // update the index
                Map<Document, Integer> docToIndex = new HashMap<>();
                if (index_map.containsKey(q)) {
                    docToIndex = index_map.get(q);
                }
                docToIndex.put(d, dataset.size());
                index_map.put(q, docToIndex);

            }
        }

        return new TestFeatures(dataset, index_map);
    }

    @Override
    public Map<Query, List<Document>> testing(TestFeatures tf, Classifier model) {
        /*
         * @TODO: Your code here
         */

        Map<Query, List<Document>> rankedDocs = new HashMap<Query, List<Document>>();

        for (Query q : tf.index_map.keySet()) {
            List<Pair<Document, Double>> rankedPairs = new ArrayList<Pair<Document, Double>>();
            for (Document d : tf.index_map.get(q).keySet()) {
                Map<Document, Integer> map = tf.index_map.get(q);
                Integer index = map.get(d);
                Instance instance = tf.features.get(index-1);
                double relevance = 0.0;
                try {
                    relevance = model.classifyInstance(instance);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                rankedPairs.add(new Pair<>(d, relevance));
            }

            // sort (doc,rel) pairs
            Collections.sort(rankedPairs, (p1, p2) -> {
                Double rel1 = p1.getSecond();
                Double rel2 = p2.getSecond();
                return rel2.compareTo(rel1);
            });

            // loop and put in rankedDocs ...
            List<Document> documents = new ArrayList<>();
            for (int i=0; i<rankedPairs.size(); ++i) {
                Document document = rankedPairs.get(i).getFirst();
                documents.add(document);
            }

            // put the mapping
            rankedDocs.put(q, documents);

        }

        return rankedDocs;
    }




}
