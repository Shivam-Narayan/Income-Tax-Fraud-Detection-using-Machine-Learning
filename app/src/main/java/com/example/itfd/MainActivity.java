package com.example.itfd;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {
    EditText ed1, ed2;
    String username,password;
    SharedPreferences sharedPreferences;
    SharedPreferences.Editor editor;
    private static final String TAG = "MainActivity";
    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ed1 = findViewById(R.id.ed1);
        ed2 = findViewById(R.id.ed2);
        sharedPreferences = getSharedPreferences("mypref",MODE_PRIVATE);
        editor = sharedPreferences.edit();
    }

    public void Login(View view) {
        try {
            username = ed1.getText().toString();
            password = ed2.getText().toString();
            if(username.equals("rafeeda") && password.equals("rafeeda")){
                editor.putString("name",username);
                editor.putString("password",password);
                editor.commit();
                Intent intent = new Intent(this,PredictionActivity.class);
                startActivity(intent);
            }
            else{
                Toast.makeText(this,"Login Unsuccessful",Toast.LENGTH_LONG).show();
            }

        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
            Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }

    }
}
