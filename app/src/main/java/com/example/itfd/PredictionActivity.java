package com.example.itfd;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.Toast;
public class PredictionActivity extends AppCompatActivity {
    private EditText age, hourPerWeek;
    private AutoCompleteTextView education, workClass;
    private Spinner occupation, maritalStatus, race, nativeCountry, declaredIncome, actualIncome;
    RadioGroup radioGroup;
    RadioButton male, female;
    String gender;
    private static final String TAG = "PredictionActivity";
    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_prediction);
        // Initialize the spinners and the button
        age = findViewById(R.id.ed_1);
        hourPerWeek = findViewById(R.id.ed_2);
        education = findViewById(R.id.auto_1);
        workClass = findViewById(R.id.auto_2);
        occupation = findViewById(R.id.spinner1);
        maritalStatus = findViewById(R.id.spinner2);
        race = findViewById(R.id.spinner3);
        nativeCountry = findViewById(R.id.spinner4);
        declaredIncome = findViewById(R.id.spinner5);
        actualIncome = findViewById(R.id.spinner6);
        radioGroup = findViewById(R.id.rg);
        male = findViewById(R.id.rb1);
        female = findViewById(R.id.rb2);

        // Set the options for autoCompleteTextView and Spinners
        ArrayAdapter<CharSequence> educationAdapter = ArrayAdapter.createFromResource(this, R.array.Education, android.R.layout.simple_spinner_item);
        educationAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        education.setAdapter(educationAdapter);
        ArrayAdapter<CharSequence> workclassAdapter = ArrayAdapter.createFromResource(this, R.array.WorkClass, android.R.layout.simple_spinner_item);
        workclassAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        workClass.setAdapter(workclassAdapter);
        ArrayAdapter<CharSequence> occupationAdapter = ArrayAdapter.createFromResource(this, R.array.Occupation, android.R.layout.simple_spinner_item);
        occupationAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        occupation.setAdapter(occupationAdapter);
        ArrayAdapter<CharSequence> maritalstatusAdapter = ArrayAdapter.createFromResource(this, R.array.MaritalStatus, android.R.layout.simple_spinner_item);
        maritalstatusAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        maritalStatus.setAdapter(maritalstatusAdapter);
        ArrayAdapter<CharSequence> raceAdapter = ArrayAdapter.createFromResource(this, R.array.Race, android.R.layout.simple_spinner_item);
        raceAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        race.setAdapter(raceAdapter);
        ArrayAdapter<CharSequence> nativecountryAdapter = ArrayAdapter.createFromResource(this, R.array.NativeCountry, android.R.layout.simple_spinner_item);
        nativecountryAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        nativeCountry.setAdapter(nativecountryAdapter);
        ArrayAdapter<CharSequence> declaredincomeAdapter = ArrayAdapter.createFromResource(this, R.array.DeclaredIncome, android.R.layout.simple_spinner_item);
        declaredincomeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        declaredIncome.setAdapter(declaredincomeAdapter);
        ArrayAdapter<CharSequence> actualincomeAdapter = ArrayAdapter.createFromResource(this, R.array.ActualIncome, android.R.layout.simple_spinner_item);
        actualincomeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        actualIncome.setAdapter(actualincomeAdapter);
    }
    public void Predict(View view) {
        // Get the values selected in the spinners
        int Age = Integer.parseInt(age.getText().toString());
        String Workclass = workClass.getText().toString();
        String Education = education.getText().toString();
        String Maritalstatus = maritalStatus.getSelectedItem().toString();
        String Occupation = occupation.getSelectedItem().toString();
        String RACE = race.getSelectedItem().toString();
        String Nativecountry = nativeCountry.getSelectedItem().toString();
        int Hoursperweek = Integer.parseInt(hourPerWeek.getText().toString());
        String Declaredincome = declaredIncome.getSelectedItem().toString();
        String Actualincome = actualIncome.getSelectedItem().toString();

        if(male.isChecked()){
            gender = male.getText().toString();
        }
        else if(female.isChecked()){
            gender = female.getText().toString();
        }

        // Preprocess the input values
        float[] input = preprocessInput(Age, Workclass, Education, Maritalstatus, Occupation, RACE, gender, Hoursperweek,   Nativecountry, Declaredincome, Actualincome);

        try {
            float[] output = new Classifier(getAssets(), "best_lr_model.tflite", new int[]{1, 11}, new int[]{1, 1}).classify(input);
            // Get the predicted income class
            int predictedClass;
            if (Declaredincome.equals("<=50K") && Actualincome.equals(">50K")) {
                predictedClass = output[0] > 0.5 ? 1 : 0;
            } else {
                predictedClass = output[0] <= 0.5 ? 1 : 0;
            }

            // Pass the predicted class value to the DisplayActivity
            Intent intent = new Intent(this, DisplayActivity.class);
            intent.putExtra("predictedClass", predictedClass);
            startActivity(intent);
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
            Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private float[] preprocessInput(int Age, String Workclass, String Education, String Martialstatus, String Occupation, String RACE, String gender, int Hoursperweek, String Nativecountry, String Declaredincome, String Actualincome) {
        // Perform any necessary preprocessing on the input values
        // Here, we will convert the categorical variables to one-hot encoded vectors


        float[] input = new float[29];
        input[0] = Age;
        input[1] = Workclass.equals("Private") ? 1 : 0;
        input[2] = Workclass.equals("Self-emp-not-inc") ? 1 : 0;
        input[3] = Workclass.equals("Local-gov") ? 1 : 0;
        input[4] = Workclass.equals("Federal-gov") ? 1 : 0;
        input[5] = Education.equals("Bachelors") ? 1 : 0;
        input[6] = Education.equals("Masters") ? 1 : 0;
        input[7] = Education.equals("Doctorate") ? 1 : 0;
        input[8] = Education.equals("HS-grad") ? 1 : 0;
        input[9] = Martialstatus.equals("Never-married") ? 1 : 0;
        input[10] = Martialstatus.equals("Married-civ-spouse") ? 1 : 0;
        input[11] = Martialstatus.equals("Divorced") ? 1 : 0;
        input[12] = Martialstatus.equals("Separated") ? 1 : 0;
        input[13] = Occupation.equals("Exec-managerial") ? 1 : 0;
        input[14] = Occupation.equals("Tech-support") ? 1 : 0;
        input[15] = Occupation.equals("Sales") ? 1 : 0;
        input[16] = Occupation.equals("Other-service") ? 1 : 0;
        input[17] = RACE.equals("Black") ? 1 : 0;
        input[18] = RACE.equals("White") ? 1 : 0;
        input[19] = RACE.equals("Asian-Pac-Islander") ? 1 : 0;
        input[20] = RACE.equals("Amer-Indian-Eskimo") ? 1 : 0;
        input[21] = gender.equals("Male") ? 1 : 0;
        input[22] = Hoursperweek;
        input[23] = Nativecountry.equals("United-States") ? 1 : 0;
        input[24] = Nativecountry.equals("Japan") ? 1 : 0;
        input[25] = Nativecountry.equals("India") ? 1 : 0;
        input[26] = Nativecountry.equals("Germany") ? 1 : 0;
        input[27] = Declaredincome.equals("<=50K") ? 1 : 0;
        input[28] = Actualincome.equals(">50K") ? 1 : 0;
        return input;
    }

    public void Back(View view) {
        Intent intent = new Intent(this, MainActivity.class);
        startActivity(intent);
    }
}