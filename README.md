# CS Round Winner Predictor

This project predicts the winner of a Counter-Strike: Global Offensive (CSGO) round based on various in-game attributes and statistics using three different machine learning models. The models used are K-Nearest Neighbors (KNN), Random Forest, and a neural network (optional). Additionally, the project visualizes the data using histograms and heatmaps and evaluates the performance of each model.

## Installation

**Clone this repository:** git clone https://github.com/yourusername/csgo-round-winner-predictor.git

**Navigate to the project directory:** cd csgo-round-winner-predictor

**Install the required packages**

**Run the main Python script:** python main.py

## Data
The dataset used in this project is sourced from OpenML. It contains various attributes and statistics from CS rounds.

The data is downloaded and processed directly from the provided URL.

Non-essential lines in the dataset (e.g., comments and metadata) are removed, leaving only the relevant game statistics.

## Models Used
**1. K-Nearest Neighbors (KNN)**

The KNN model is first trained using default parameters.

A grid search with cross-validation is performed to optimize the hyperparameters.

The model's accuracy is evaluated before and after hyperparameter tuning.

**2. Random Forest**

The Random Forest classifier is used as the second model.

The model is trained on the scaled training data, and its accuracy is evaluated.

**3. Neural Network (Optional)**

The project can be extended by implementing a neural network using TensorFlow/Keras to predict the round winner.

## Visualization
The project includes visualizations to better understand the data and the model's performance:

**Heatmap:** A heatmap is generated to visualize the correlation between the selected features and the target variable (t_win).

**Histograms:** Histograms are created for each selected feature to show the distribution of data.

## Results
The accuracy of each model is printed after training:

**KNN:** Accuracy before and after hyperparameter tuning.

**Random Forest:** Accuracy after training.

These results can be used to determine which model performs best for predicting CS round winners.
