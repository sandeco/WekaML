import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ClassifyAInstance {

	
	static Attribute curso;
	static Attribute ano;		
	static Attribute sexo;
	static Attribute origem;
	static Attribute anoMat;
	static Attribute disciplina;
	static Attribute N1;
	static Attribute F1;
    
	static Attribute status;
	static ArrayList<String> statusList = new ArrayList<>();
	
	public static void main(String[] args) {

		try {
		
			DataSource source = new DataSource("c:/ARFF/todos1.arff");
			
			Instances train = source.getDataSet();			
			train.setClassIndex(train.numAttributes()-1);
			
			NaiveBayes nb = new NaiveBayes();
			nb.buildClassifier(train);
			
			
			Instance instance = newInstance();
			
			double result = nb.classifyInstance(instance);			
			String output = statusList.get((int)result); 
			
			System.out.println(output);
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}


	}


	public static Instance newInstance(){
		
		
		Instances virtualArff = new Instances("TestInstances",getListAttributes(),0);
		virtualArff.setClassIndex(virtualArff.numAttributes() - 1);
		
		
		// Create empty instance with three attribute values 
		Instance inst = new DenseInstance(virtualArff.numAttributes());
        
		virtualArff.add(inst);
        
        
		inst.setValue(curso, 1);
		inst.setValue(ano, 1996);
		
		inst.setValue(origem, 1);
		inst.setValue(anoMat, 4);
		inst.setValue(disciplina, 1);
		inst.setValue(N1, 0.0);
		inst.setValue(F1, 5);
		inst.setMissing(status);
		
		
		
		
		return virtualArff.firstInstance();
		
	}
	
	
	public static ArrayList<Attribute> getListAttributes(){

		curso = new Attribute("CURSO");
		ano   = new Attribute("ANO");		
		
		//atributo string
		ArrayList<String> sexoList = new ArrayList<>();
        sexoList.add("F");
        sexoList.add("M");
        sexo = new Attribute("SEXO",sexoList);
        
        origem = new Attribute("ORIGEM");
		anoMat = new Attribute("ANO_MAT");
		disciplina = new Attribute("DESC_DISCIPLINA");
		N1 = new Attribute("N1");
		F1 = new Attribute("F1");
		
		
		
        statusList.add("A");
        statusList.add("R");

		status = new Attribute("STATUS",statusList);
		
		
		ArrayList<Attribute> attributeList = new ArrayList<>();
		
		attributeList.add(curso);
		attributeList.add(ano);
        attributeList.add(sexo);
		attributeList.add(origem);
		attributeList.add(anoMat);
		attributeList.add(disciplina);
		attributeList.add(N1);
		attributeList.add(F1);	
        attributeList.add(status);
		
        return attributeList;
	
	}


}
