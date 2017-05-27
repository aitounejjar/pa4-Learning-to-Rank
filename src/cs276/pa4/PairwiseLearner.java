package cs276.pa4;

import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.SelectedTag;

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
    public Instances extractTrainFeatures(String train_data_file,
                                          String train_rel_file, Map<String, Double> idfs) {
    /*
     * @TODO: Your code here:
     * Get signal file 
     * Construct output dataset of type Instances
     * Add new attribute  to store relevance in the train dataset
     * Populate data
     */
        return null;
    }

    @Override
    public Classifier training(Instances dataset) {
    /*
     * @TODO: Your code here
     * Build classifer
     */
        return null;
    }

    @Override
    public TestFeatures extractTestFeatures(String test_data_file,
                                            Map<String, Double> idfs) {
    /*
     * @TODO: Your code here
     * Use this to build the test features that will be used for testing
     */
        return null;
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
