import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Weather {

	private Attribute outlook;
	private Attribute temperature;		
	private Attribute humidity;
	private Attribute windy;
	private Attribute play;
	
	private ArrayList<Attribute> attributeList = new ArrayList<>();
	
	private ArrayList<String> outlookOptions = new ArrayList<String>();
	private ArrayList<String> playOptions = new ArrayList<String>();
	private ArrayList<String> windOptions = new ArrayList<String>();
	
	
	public Weather(){
		outlookOptions.add(WeatherStrings.SUNNY);
		outlookOptions.add(WeatherStrings.OVERCAST);
		outlookOptions.add(WeatherStrings.RAINY);
	
		outlook = new Attribute(WeatherStrings.OUTLOOK,outlookOptions);
		
		
		playOptions.add("yes");
		playOptions.add("no");
		
		play = new Attribute(WeatherStrings.PLAY,playOptions);
		
		windOptions.add(WeatherStrings.TRUE);
		windOptions.add(WeatherStrings.FALSE);
		
		windy = new Attribute(WeatherStrings.WINDY,windOptions);
		
		humidity = new Attribute(WeatherStrings.HUMIDITY);
		temperature = new Attribute(WeatherStrings.TEMPERATURE);
		
		attributeList.add(outlook);
		attributeList.add(temperature);
		attributeList.add(humidity);
		attributeList.add(windy);
		attributeList.add(play);	
		
	}
	
	
	public void classify() {
		try {
			
			
			DataSource source = new DataSource("c:/ARFF/weather.arff");
			
			Instances train = source.getDataSet();			
			train.setClassIndex(train.numAttributes()-1);	
			
			
			//weka.classifiers.lazy.IBk nb = new IBk();
			
			NaiveBayes nb = new NaiveBayes();
			
			nb.buildClassifier(train);

			
			
			
			/*VIRTUAL ARFF*/
			Instances virtualArff = new Instances("TestInstances",attributeList,0);
			virtualArff.setClassIndex(virtualArff.numAttributes() - 1);
			
			
			// Create empty instance with three attribute values 
			Instance inst = new DenseInstance(virtualArff.numAttributes());
			virtualArff.add(inst);
	        
	        
			inst.setValue(outlook,WeatherStrings.SUNNY);
			inst.setValue(temperature, 60);
			inst.setValue(humidity, 95);
			inst.setValue(windy, WeatherStrings.FALSE);
			inst.setMissing(play);			
			
			
			inst.setDataset(train);

			double result = nb.classifyInstance(inst);
			
			String output = playOptions.get((int)result); 
			
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

}
