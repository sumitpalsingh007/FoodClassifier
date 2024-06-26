package com.helloworldtech;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.translate.TranslateException;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class FoodMacroCalculator {

    public static void main(String[] args) {
        try {
            // Load image
            Path imagePath = Paths.get("/Users/sps/Downloads/haldiram.jpg");
            Image img = ImageFactory.getInstance().fromFile(imagePath);

            // Define the class names
            List<String> classNames = Arrays.asList(NutritionInfo.class.getName());

            // Define the translator
            Translator<Image, Classifications> translator = new MyTranslator(classNames);

            // Define criteria with the local model path
            Criteria<Image, Classifications> criteria = Criteria.builder()
                    .setTypes(Image.class, Classifications.class)
                    .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                    .optModelUrls("https://storage.googleapis.com/tfhub-modules/google/aiy/vision/classifier/food_V1/1.tar.gz")
                    .optTranslator(translator)
                    .optProgress(new ProgressBar())
                    .optDevice(Device.cpu())
                    .build();

            // Load the nutrition database
            Map<String, NutritionInfo> nutritionDatabase = loadNutritionDatabase("src/main/resources/nutrition_database.csv");

            // Load the model and predict
            try (Model model = ModelZoo.loadModel(criteria)) {
                Classifications classifications = model.newPredictor(translator).predict(img);

                double totalCarbs = 0;
                double totalProtein = 0;
                double totalFats = 0;
                double totalFiber = 0;

                for (Classifications.Classification classification : classifications.items()) {
                    String foodItem = classification.getClassName().toLowerCase();
                    if (nutritionDatabase.containsKey(foodItem)) {
                        NutritionInfo info = nutritionDatabase.get(foodItem);
                        totalCarbs += info.carbs;
                        totalProtein += info.protein;
                        totalFats += info.fats;
                        totalFiber += info.fiber;
                    }
                }

                System.out.println("Total Carbs: " + totalCarbs);
                System.out.println("Total Protein: " + totalProtein);
                System.out.println("Total Fats: " + totalFats);
                System.out.println("Total Fiber: " + totalFiber);
            }

        } catch (IOException | ModelException | TranslateException e) {
            e.printStackTrace();
        }
    }

    private static Map<String, NutritionInfo> loadNutritionDatabase(String csvFilePath) throws IOException {
        Map<String, NutritionInfo> nutritionDatabase = new HashMap<>();

        try (Reader reader = Files.newBufferedReader(Paths.get(csvFilePath));
             CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader())) {

            for (CSVRecord record : csvParser) {
                String foodItem = record.get("FoodItem").toLowerCase();
                double carbs = Double.parseDouble(record.get("Carbs"));
                double protein = Double.parseDouble(record.get("Protein"));
                double fats = Double.parseDouble(record.get("Fats"));
                double fiber = Double.parseDouble(record.get("Fiber"));

                nutritionDatabase.put(foodItem, new NutritionInfo(carbs, protein, fats, fiber));
            }
        }

        return nutritionDatabase;
    }
}

class MyTranslator implements Translator<Image, Classifications> {

    private List<String> classNames;

    public MyTranslator(List<String> classNames) {
        this.classNames = classNames;
    }

    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        NDArray probabilities = list.singletonOrThrow();
        return new Classifications(classNames, probabilities);
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        //Apply transformations directly on the Image object
        input.resize(224, 224, true)/*.add(new ToTensor())*/;
        return new NDList(input.toNDArray(ctx.getNDManager()));
    }

    @Override
    public Batchifier getBatchifier() {
        return null;
    }
}

class NutritionInfo {
    double carbs;
    double protein;
    double fats;
    double fiber;

    public NutritionInfo(double carbs, double protein, double fats, double fiber) {
        this.carbs = carbs;
        this.protein = protein;
        this.fats = fats;
        this.fiber = fiber;
    }
}
