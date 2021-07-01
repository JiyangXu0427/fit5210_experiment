package demo;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.URL;
import java.sql.Array;
import java.sql.SQLOutput;
import java.util.Map;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.Set;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import core.explorer.ChordalysisModeller;
import core.explorer.ChordalysisModellingSMT;
import core.model.DecomposableModel;
import core.model.Inference;
import loader.LoadWekaInstances;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

public class Test_Inference_Log_Ordered_Joint_Prob {

    public static void main(String[] args) throws IOException {
        String[] filenames = new String[]{"foresttype","http"};
//        String[] filenames = new String[]{"annthyroid", "cardio", "ionosphere", "mammography","satellite", "shuttle", "thyroid"，"smtp","satimage-2","pendigits","speech"};
        String[] types = new String[]{"10BIN", "15BIN"};
        for (String datasetName : filenames) {
            for (String datatype : types) {
                //annthyroid_discretized_10BIN_withLabel
                File csvFile = new File("../eif-master/datasets/" + datasetName + "_discretized_" + datatype +"_withoutLabel.csv");

                //load the csv from temp file
                CSVLoader loader = new CSVLoader();
                loader.setFile(csvFile);
                //??what's this??
                loader.setNominalAttributes("first-last");

                //Load the dataset into the Weka Instances
                Instances instances = loader.getDataSet();
//        System.out.println(instances.toSummaryString());

                //Get the feature names, get the possible values of each feature
                //print a summary of the features names and features values.
                String[] variablesNames = new String[instances.numAttributes()];
                String[][] outcomes = new String[instances.numAttributes()][];
                for (int i = 0; i < variablesNames.length; i++) {
                    variablesNames[i] = instances.attribute(i).name();
                    outcomes[i] = new String[instances.attribute(i).numValues() + 1];//+1 for missing
                    for (int j = 0; j < outcomes[i].length - 1; j++) {
                        outcomes[i][j] = instances.attribute(i).value(j);
                    }
                    outcomes[i][outcomes[i].length - 1] = "missing";
                    System.out.println("Dom(" + variablesNames[i] + ") = " + Arrays.toString(outcomes[i]));
                }

                //Build Chordalysis model
                ChordalysisModeller.Data mydata = LoadWekaInstances.makeModelData(instances);
                ChordalysisModellingSMT modeller = new ChordalysisModellingSMT(mydata, 0.05);
//        System.out.println("Learning...");
                modeller.buildModel();

                //Get the best model order
                DecomposableModel bestModel = modeller.getModel();
//        System.out.println("The model selected is:");
                String bestModelToStringResult = bestModel.toString(variablesNames);
//        System.out.println(bestModelToStringResult);

                //Get the order of the clique, remove the duplicate but maintain the original order, then store in ArrayList
                String tmp = bestModelToStringResult.replace("[", "");
                tmp = tmp.replace("]", "");
                String[] bestModelToStringResultList = tmp.split(" ");
                ArrayList<String> bestFeatureOrder = new ArrayList<>();
                for (String name : bestModelToStringResultList) {
                    if (!name.equals("") && !bestFeatureOrder.contains(name)) {
                        bestFeatureOrder.add(name);
                    }
                }

                //Start to inference
                Inference inference = new Inference(bestModel, variablesNames, outcomes);
                //??what's this??
                inference.setProbabilities(modeller.getLattice());

                //convert the feature_name and feature_value array into list (to be used in calculating joint log prob)
                List<String> variablesNamesList = Arrays.asList(variablesNames);
                List<String[]> outComeList = Arrays.asList(outcomes);
//        System.out.println(variablesNamesList.size());
//        System.out.println(outComeList.size());

                //will be used to store the joint log prob result for each data
                ArrayList<Double> logProbs = new ArrayList<>();

                //will calculate the joint log prob for each data
                //According to chain rule, joint log prob P(V1,V2,V3,...,Vn) = P(V1)*P(V2|V1)*P(V3|V1,V2)*...*P(Vn|V1,V2,..,Vn-1)
                //For the order of the chain, we will use the order generated by Chordalysis
                for (int i = 0; i < instances.numInstances(); i++) {
                    //get one row of data
                    Instance oneRowOfData = instances.instance(i);
                    //                System.out.println(oneRowOfData.toString());

                    //used to to the joint log prob for this row of data
                    double logProb = 0;

                    //used to store evidence for calculating conditional prob
                    ArrayList<String> evidences_value = new ArrayList<>();
                    ArrayList<String> evidences_feature_name = new ArrayList<>();

                    for (String featureName : bestFeatureOrder) {
                        //find the attribute class object by name
                        Attribute att = instances.attribute(featureName);
                        //get the value of the attribute
                        String featureValue = oneRowOfData.stringValue(att);
                        //deal with missing value
                        if (featureValue.equals("?") || featureValue.equals("")) {
                            featureValue = "missing";
                        }
                        //add evidence to the inference
                        for (int index = 0; index < evidences_value.size(); index++) {
                            inference.addEvidence(evidences_feature_name.get(index), evidences_value.get(index));
                            if (index == evidences_value.size() - 1) {
                                inference.recordEvidence();
                            }
                        }
                        //store evidence for next round
                        evidences_feature_name.add(featureName);
                        evidences_value.add(featureValue);

                        //get the conditional prob for each class of the feature
                        double[] initBelief = inference.getBelief(featureName);

                        //get the conditional prob for this feature of this data
                        int index_feature_position = variablesNamesList.indexOf(featureName);
                        String[] feature_value_array = outComeList.get(index_feature_position);
                        List<String> feature_value_list = Arrays.asList(feature_value_array);
                        int index_for_this = feature_value_list.indexOf(featureValue);
                        logProb += Math.log10(initBelief[index_for_this]);

                        //clear the evidence. evidence will be added again at next round.
                        inference.clearEvidences();
                    }
                    //add the log prob result to the result list
                    logProbs.add(logProb);
                }
                //output the log prob result
                XSSFWorkbook workbook = new XSSFWorkbook();
                XSSFSheet spreadsheet = workbook.createSheet("result");
                XSSFRow row = spreadsheet.createRow(0);
                XSSFCell cell = row.createCell(0);
                cell.setCellValue("Log_prob");
                int rowCount = 1;
                for (Double logProbValue : logProbs) {
                    XSSFRow dataRow = spreadsheet.createRow(rowCount);
                    XSSFCell dataCell = dataRow.createCell(0);
                    dataCell.setCellValue(logProbValue);
                    rowCount++;
                }

                try (FileOutputStream outputStream = new FileOutputStream(new File("./chodalysis_result/" + datasetName + "-" + datatype + "-ordered_log_prob_result.xlsx"))) {
                    workbook.write(outputStream);
                    outputStream.close();
                    workbook.close();
                }
            }
        }
    }
}


//please disregard the code below
//                System.out.println("instance" + i + featureName);
//                if(i == 4074 && featureName.equals("stalk-root")){
//                    System.out.println(initBelief.length);
//                    System.out.println(initBelief.toString());
//                    System.out.println(index_feature_position);
//                    System.out.println(feature_value_array.toString());
//                    System.out.println(feature_value_list.toString());
//                    System.out.println(index_for_this);
//                }