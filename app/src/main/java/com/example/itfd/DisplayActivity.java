package com.example.itfd;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

public class DisplayActivity extends AppCompatActivity {
    @SuppressLint({"MissingInflatedId", "SetTextI18n"})
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_display);
        // Get the predictedClass value from the intent
        int predictedClass = getIntent().getIntExtra("predictedClass",-1);

        // Set the text of a TextView based on the predictedClass value
        TextView predictedClassTextView = findViewById(R.id.textview);
        if (predictedClass == 1) {
            predictedClassTextView.setText("Suspicious of committing Income tax fraud");
        } else {
            predictedClassTextView.setText("Not suspicious of committing Income tax fraud");
        }
    }

    public void Back(View view) {
        Intent intent = new Intent(this, PredictionActivity.class);
        startActivity(intent);
    }
}