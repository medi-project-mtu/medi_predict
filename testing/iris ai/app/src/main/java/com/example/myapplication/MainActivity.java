package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;


import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;


import com.example.myapplication.ml.Diabetes;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button res = findViewById(R.id.button);
        TextView resTV = findViewById(R.id.res);
        res.setOnClickListener(view -> {
//            0,84,82,31,125,38.2,0.233,23,0
//            7,195,70,33,145,25.1,0.163,55,1, pred:1.0
//            4,125,70,18,122,28.9,1.144,45,1, pred: 0.1219
//            4,103,60,33,192,24,0.966,33,0
//            3,163,70,18,105,31.6,0.268,28,1
//            9,171,110,24,240,45.4,0.721,54,1
//            3,129,64,29,115,26.4,0.219,28,1, pred: 0.1219
//            1,181,64,30,180,34.1,0.328,38,1
//            7,114,76,17,110,23.8,0.466,31,0
//            0,116,64,39,225,40.2,0.72,50,0
//            5,136,84,41,88,35,0.286,35,1, pred: 0.1219
//            1,111,62,13,182,24,0.138,23,0, pred: 000000000000000
//            7,150,78,29,126,35.2,0.692,54,1, pred: 000000000000000000000000
//            7,97,76,32,91,40.9,0.871,32,1
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(8*4);
            float[] kek =  {7,97,76,32,91,40.9f,0.871f,32};
            byteBuffer.putFloat(kek[0]);
            byteBuffer.putFloat(kek[1]);
            byteBuffer.putFloat(kek[2]);
            byteBuffer.putFloat(kek[3]);
            byteBuffer.putFloat(kek[4]);
            byteBuffer.putFloat(kek[5]);
            byteBuffer.putFloat(kek[6]);
            byteBuffer.putFloat(kek[7]);
            try {
                Diabetes model = Diabetes.newInstance(MainActivity.this);

                // Creates inputs for reference.
                TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 8}, DataType.FLOAT32);
                inputFeature0.loadBuffer(byteBuffer);

                // Runs model inference and gets result.
                Diabetes.Outputs outputs = model.process(inputFeature0);
                float[] outputFeature0 = outputs.getOutputFeature0AsTensorBuffer().getFloatArray();

                resTV.setText(Float.toString(outputFeature0[0]));

                // Releases model resources if no longer used.
                model.close();
            } catch (IOException e) {
                // TODO Handle the exception
            }
        });
    }
}