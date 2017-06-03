package cs276.pa3;

import cs276.pa4.Document;
import cs276.pa4.Query;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;

import static cs276.pa4.Util.UNSEEN_TERM_ID;

/**
 * Skeleton code for the implementation of a
 * Cosine Similarity Scorer in Task 1.
 */
public class CosineSimilarityScorer extends AScorer {

    /*
     * TODO: You will want to tune the values for
     * the weights for each field.
     */

    final double urlweight            = 10;
    final double titleweight          = 10;
    final double headerweight         = 7;
    final double anchorweight         = 8;
    final double bodyweight           = .6;
    final double smoothingBodyLength  = 15000;

    /**
     * Construct a Cosine Similarity Scorer.
     *
     * @param idfs the map of idf values
     */
    public CosineSimilarityScorer(Map<String, Double> idfs) {
        super(idfs);



    }


    /**
     * Get the net score for a query and a document.
     *
     * @param tfs     the term frequencies
     * @param q       the Query
     * @param tfQuery the term frequencies for the query
     * @param d       the Document
     * @return the net score
     */
    public double getNetScore(Map<String, Map<String, Double>> tfs, Query q, Map<String, Double> tfQuery, Document d) {
        double score = 0.0;

        /*
         * TODO : Your code here
         * See Equation 2 in the handout regarding the net score between "a query vector" and "the term score vector for the document".
         *
         */


        double queryLength = 0.0;
        double documentLength = 0.0;


        for (String term : q.queryWords) {

            // compute the tf-idf weight of the term (weight = tf x idf)
            double tf = tfQuery.get(term);//1 + Math.log(tfQuery.get(term));
            double idf = idfs.containsKey(term) ? idfs.get(term) : idfs.get(UNSEEN_TERM_ID);
            double tfIdfWeight = tf * idf;
            queryLength += Math.pow(tfIdfWeight, 2);

            // loop over sections: "url", "title", "body", "header", "anchor"
            double sectionScores = 0.0;
            for (String section : tfs.keySet()) {
                double sectionWeight = getSectionWeight(section);
                double sectionScore = sectionWeight * (tfs.get(section).containsKey(term) ? tfs.get(section).get(term) : 0.0);
                sectionScores += sectionScore;
            } // end inner for loop

            // update the document length
            documentLength += Math.pow(sectionScores, 2);

            // do the multiplication in equation (2)
            score += (tfIdfWeight * sectionScores);

        } // end outer for loop

        return score;
    }

    /**
     * Normalize the term frequencies.
     *
     * @param tfs the term frequencies
     * @param d   the Document
     * @param q   the Query
     */
    public void normalizeTFs(Map<String, Map<String, Double>> tfs, Document d, Query q) {

        /*
         * TODO : Your code here
         * Note that we should give uniform normalization to all
         * fields as discussed in the assignment handout.
         */

        double bodyLengthSmoother = d.body_length + smoothingBodyLength;

        for (String tfType : tfs.keySet()) {
            Map<String, Double> map = tfs.get(tfType);
            for (String w : map.keySet()) {
                double adjusted = Math.log10(1+map.get(w)) / bodyLengthSmoother;
                map.put(w, adjusted);
            }
        }
    }

    /**
     * Write the tuned parameters of cosineSimilarity to file.
     * Only used for grading purpose, you should NOT modify this method.
     *
     * @param filePath the output file path.
     */
    private void writeParaValues(String filePath) {
        try {
            File file = new File(filePath);
            if (!file.exists()) {
                file.createNewFile();
            }
            FileWriter fw = new FileWriter(file.getAbsoluteFile());
            String[] names = {
                    "urlweight", "titleweight", "bodyweight", "headerweight",
                    "anchorweight", "smoothingBodyLength"
            };
            double[] values = {
                    this.urlweight, this.titleweight, this.bodyweight,
                    this.headerweight, this.anchorweight, this.smoothingBodyLength
            };
            BufferedWriter bw = new BufferedWriter(fw);
            for (int idx = 0; idx < names.length; ++idx) {
                bw.write(names[idx] + " " + values[idx]);
                bw.newLine();
            }
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    /** Get the similarity score between a document and a query.
     * @param d the Document
     * @param q the Query
     * @return the similarity score.
     */
    public double getSimScore(Document d, Query q) {

        // get document term frequencies
        Map<String, Map<String, Double>> docTermFreqs = this.getDocTermFreqs(d, q);

        // normalize the document term frequencies
        this.normalizeTFs(docTermFreqs, d, q);

        // get query term frequencies, no normalization needed for these
        Map<String, Double> tfQuery = getQueryFreqs(q);

        // Write out tuned cosineSimilarity parameters
        // This is only used for grading purposes.
        // You should NOT modify the writeParaValues method.
        writeParaValues("cosinePara.txt");

        return getNetScore(docTermFreqs, q, tfQuery, d);
    }

    private double getSectionWeight(String section) {
        double sectionWeight;
        switch (section) {
            case "url"      : sectionWeight = urlweight;    break;
            case "title"    : sectionWeight = titleweight;  break;
            case "body"     : sectionWeight = bodyweight;   break;
            case "header"   : sectionWeight = headerweight; break;
            case "anchor"   : sectionWeight = anchorweight; break;
            default         : throw new RuntimeException("Illegal section type of '" + section + "' was found in the tfs map.");
        }
        return sectionWeight;
    }




}
