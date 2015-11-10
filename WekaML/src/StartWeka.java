import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class StartWeka {

	public static void main(String[] args) {		
				
		try {
			
			DataSource source = new DataSource("c:/ARFF/todos1.arff");
			
			Instances train = source.getDataSet();			
			train.setClassIndex(train.numAttributes()-1);
			
			NaiveBayes nb = new NaiveBayes();
			nb.buildClassifier(train);
			
			
			Evaluation ev = new Evaluation(train);
			ev.crossValidateModel(nb, train, 10, new Random(1));
			
			System.out.println(ev.toSummaryString());
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	

}
