package com.example.demo;

import org.springframework.stereotype.Service;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.InputStream;
import java.util.ArrayList;

@Service
public class WekaModelService {

    private final Classifier model;
    private final Instances datasetStructure;

    public WekaModelService() throws Exception {
        // Cargar el modelo
        try (InputStream modelStream = getClass().getResourceAsStream("/maintenance.model")) {
            if (modelStream == null) {
                throw new RuntimeException("No se encontró el archivo 'maintenance.model' en resources");
            }
            this.model = (Classifier) SerializationHelper.read(modelStream);
        }

        // Definir estructura del dataset
        ArrayList<Attribute> attributes = new ArrayList<>();

        // 1️⃣ Vehicle_Model
        ArrayList<String> models = new ArrayList<>();
        models.add("Bus");
        attributes.add(new Attribute("Vehicle_Model", models));

        // 2️⃣ Mileage
        attributes.add(new Attribute("Mileage"));

        // 3️⃣ Maintenance_History
        ArrayList<String> maintHist = new ArrayList<>();
        maintHist.add("Poor");
        maintHist.add("Average");
        maintHist.add("Good");
        attributes.add(new Attribute("Maintenance_History", maintHist));

        // 4️⃣ Reported_Issues
        attributes.add(new Attribute("Reported_Issues"));

        // 5️⃣ Vehicle_Age
        attributes.add(new Attribute("Vehicle_Age"));

        // 6️⃣ Fuel_Type
        ArrayList<String> fuelType = new ArrayList<>();
        fuelType.add("Electric");
        fuelType.add("Petrol");
        fuelType.add("Diesel");
        attributes.add(new Attribute("Fuel_Type", fuelType));

        // 7️⃣ Transmission_Type
        ArrayList<String> trans = new ArrayList<>();
        trans.add("Automatic");
        trans.add("Manual");
        attributes.add(new Attribute("Transmission_Type", trans));

        // 8️⃣ Engine_Size
        attributes.add(new Attribute("Engine_Size"));

        // 9️⃣ Last_Service_Date
        attributes.add(new Attribute("Last_Service_Date"));

        // 🔟 Service_History
        attributes.add(new Attribute("Service_History"));

        // 11️⃣ Accident_History
        attributes.add(new Attribute("Accident_History"));

        // 12️⃣ Fuel_Efficiency
        attributes.add(new Attribute("Fuel_Efficiency"));

        // 13️⃣ Tire_Condition
        ArrayList<String> tire = new ArrayList<>();
        tire.add("New");
        tire.add("Good");
        tire.add("Worn Out");
        attributes.add(new Attribute("Tire_Condition", tire));

        // 14️⃣ Brake_Condition
        ArrayList<String> brake = new ArrayList<>();
        brake.add("Good");
        brake.add("Worn Out");
        brake.add("New");
        attributes.add(new Attribute("Brake_Condition", brake));

        // 15️⃣ Battery_Status
        ArrayList<String> battery = new ArrayList<>();
        battery.add("Weak");
        battery.add("New");
        battery.add("Good");
        attributes.add(new Attribute("Battery_Status", battery));

        // 16️⃣ Clase: Need_Maintenance
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("Yes");
        classValues.add("No");
        attributes.add(new Attribute("Need_Maintenance", classValues));

        datasetStructure = new Instances("PredictionData", attributes, 0);
        datasetStructure.setClassIndex(datasetStructure.numAttributes() - 1);
    }

    public String predict(weka.core.Instance instance) throws Exception {
        double predIndex = model.classifyInstance(instance);
        return datasetStructure.classAttribute().value((int) predIndex);
    }

    public Instances getDatasetStructure() {
        return datasetStructure;
    }
}
