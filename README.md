# 50.039_FlightDelayPredictor
Group 18's project for 50.039 Theory and Practice of Deep Learning


## Model Architecture

The `ImprovedFlightDelayModel` is a neural network architecture designed for predicting flight delays by combining both static flight information and sequential weather data. This model leverages multiple advanced deep learning techniques to enhance prediction accuracy. The model incorporates:

- Bidirectional LSTM: Processes sequential weather data to capture temporal patterns and dependencies in weather conditions leading up to flights
- Attention Mechanism: Focuses on the most relevant time steps in the weather sequence that contribute to flight delays
- Deep Feature Processing: Handles static flight information through multiple fully-connected layers with regularization
- Feature Fusion: Combines both static and weather-derived features for comprehensive prediction

**Key Components:**

1. **Weather Sequence Processing:**

    Multi-layer LSTM with dropout for regularization
    
    Attention mechanism to identify critical weather patterns


2. **Static Feature Processing:**

    Multi-layer neural network with ReLU activations

    Batch normalization to stabilize learning

    Dropout layers to prevent overfitting


3. **Combined Prediction Network:**

    Integrates both feature types for final delay prediction

    Progressively reduces dimensionality to produce a single delay value

## Training the model

First, download the '2016-2021.csv' and '2016-2021_weather.csv' and place them in the same folder as the lstm+attention_final.ipynb notebook.

[Download link](https://drive.google.com/drive/folders/1FBSiQEvIoivnwkyswmi7aRI7V1qwlld5?usp=sharing)

The 2016-2021.csv contains all flight details at JFK airport of Delta Airlines from 2016 to 2021, including scheduled departure time and actual departure time.

The 2016-2021_weather.csv contains different weather data at JFK airport.

To train the model, simply run through all cells in the notebook. 

## Evaluation

Evaluation of the model using three different metrics:

- Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual delays
- Root Mean Squared Error (RMSE): Emphasizes larger prediction errors
- R² Score: Indicates the proportion of variance in delays that is predictable from the features
