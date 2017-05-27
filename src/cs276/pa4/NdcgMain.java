package cs276.pa4;

import java.io.*;
import java.util.*;

public class NdcgMain {
    Map<String, Map<String, Double>> relevScores = null;
    Map<String, List<Pair<String, Document>>> queryRankings = null;
    Map<String, Double> queryNdcg = null;


    public NdcgMain(String relFile) throws IOException {
        relevScores = new HashMap<String, Map<String, Double>>();

        // read the relevance score
        BufferedReader br = new BufferedReader(new FileReader(relFile));
        Map<String, Double> urlScores = null;
        String strLine;
        String query = "";
        while ((strLine = br.readLine()) != null) {
            if (strLine.trim().charAt(0) == 'q') {
                query = strLine.substring(strLine.indexOf(":") + 1).trim();
                urlScores = new HashMap<String, Double>();
                relevScores.put(query, urlScores);
            } else {
                String[] tokens = strLine.substring(strLine.indexOf(":") + 1).trim().split(" ");
                String url = tokens[0].trim();
                double relevance = Double.parseDouble(tokens[1]);
                if (relevance < 0)
                    relevance = 0;
                if (urlScores != null)
                    urlScores.put(url, relevance);
            }
        }
        br.close();
    }

    public double score(String rankedFile) throws IOException {
        queryRankings = new HashMap<String, List<Pair<String, Document>>>();
        queryNdcg = new HashMap<String, Double>();

        // read the query rankings and do the ndcg score calculation
        BufferedReader br = new BufferedReader(new FileReader(rankedFile));
        ArrayList<Double> rels = new ArrayList<Double>();
        List<Pair<String, Document>> urlDocs = null;
        String strLine;
        String query = "";
        Document doc = null;
        String url = "";

        int totalQueries = 0;
        double totalSum = 0;

        while ((strLine = br.readLine()) != null) {
            if (strLine.trim().charAt(0) == 'q') {
                if (rels.size() > 0) {
                    double ndcgQuery = getNdcgQuery(rels);
                    queryNdcg.put(query, ndcgQuery);
                    totalSum += ndcgQuery;
                    rels.clear();
                }
                query = strLine.substring(strLine.indexOf(":") + 1).trim();
                urlDocs = new ArrayList<Pair<String, Document>>();
                queryRankings.put(query, urlDocs);

                totalQueries++;
            } else if (strLine.trim().charAt(0) == 'u') {
                url = strLine.substring(strLine.indexOf(":") + 1).trim();
                doc = new Document(url);
                if (relevScores.containsKey(query) && relevScores.get(query).containsKey(url)) {
                    double relevance = relevScores.get(query).get(url);
                    rels.add(relevance);
                } else {
                    System.err.printf("Warning. Cannot find query %s with url %s. Ignoring this line \n", query, url);
                }
            } else if (strLine.trim().charAt(0) == 't') {
                doc.title = strLine.substring(strLine.indexOf(":") + 1).trim();
            } else if (strLine.trim().charAt(0) == 'd') {
                doc.debugStr = strLine.substring(strLine.indexOf(":") + 1).trim();
                urlDocs.add(new Pair<String, Document>(url, doc));
            }
        }
        br.close();

        if (rels.size() > 0) {
            double ndcgQuery = getNdcgQuery(rels);
            queryNdcg.put(query, ndcgQuery);
            totalSum += ndcgQuery;
        }
        //System.out.println(totalSum/totalQueries);  //print the ndcg score for your rankings
        double ndcgscore = totalSum / totalQueries;
        return ndcgscore;
    }

    public void writeOutputToFile(String outputFile) {
        // write side by side output to file such that you can compare the experiment with others
        try {
            File file = new File(outputFile);
            if (!file.exists()) {
                file.createNewFile();
            }

            FileWriter fw = new FileWriter(file.getAbsoluteFile());
            BufferedWriter bw = new BufferedWriter(fw);

            for (String curr : queryRankings.keySet()) {
                String queryStr = "query: " + curr + "\n";
                bw.write(queryStr);

                Double ndcgScore = queryNdcg.get(curr);
                String ndcgStr = "ndcg: " + ndcgScore.toString() + "\n";
                bw.write(ndcgStr);

                for (Pair<String, Document> urlDoc : queryRankings.get(curr)) {
                    Double rating = relevScores.get(curr).get(urlDoc.getFirst());
                    Document document = urlDoc.getSecond();
                    String urlString =
                            "  url: " + document.url + "\n" +
                                    "    rating: " + rating.toString() + "\n" +
                                    "    title: " + document.title + "\n" +
                                    "    debug: " + document.debugStr + "\n";
                    bw.write(urlString);
                }
            }

            bw.close();
            fw.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static double getNdcgQuery(ArrayList<Double> rels) {
        double localSum = 0, sortedSum = 0;
        for (int i = 0; i < rels.size(); i++)
            localSum += (Math.pow(2, rels.get(i)) - 1) / (Math.log(1 + i + 1) / Math.log(2));
        Collections.sort(rels, Collections.reverseOrder());
        for (int i = 0; i < rels.size(); i++)
            sortedSum += (Math.pow(2, rels.get(i)) - 1) / (Math.log(1 + i + 1) / Math.log(2));
        if (sortedSum == 0)
            return 1;
        else
            return localSum / sortedSum;
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 3) {
            System.out.println(
                    "Please specify 3 files: " +
                            "(i) the ranked input file and " +
                            "(ii) the input file containing the relevance scores " +
                            "(iii) the output file"
            );
            System.exit(1);
        }
        String rankedFile = args[0];
        String relFile = args[1];
        NdcgMain ndcg = new NdcgMain(relFile);
        double ndcg_score = ndcg.score(rankedFile);
        System.err.println(ndcg_score);
        ndcg.writeOutputToFile(args[2]);
    }
}
