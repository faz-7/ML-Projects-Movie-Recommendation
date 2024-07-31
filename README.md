# Movie Recommendation System

## Overview
This repository implements a movie recommendation system using TensorFlow and Keras. The model utilizes collaborative filtering techniques to predict user ratings for movies and recommend top-rated films based on user preferences.

## Table of Contents
- [Introduction](#introduction)
- [Data Description](#data-description)
- [Techniques Used](#techniques-used)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Recommendations](#recommendations)
- [Conclusion](#conclusion)

## Introduction
The MovieLens dataset is a popular dataset used for testing and evaluating recommendation algorithms. This project aims to build a recommendation system that predicts movie ratings and suggests movies to users based on their past ratings.

## Data Description
The dataset used in this project is the MovieLens 100K dataset, which contains:
- **User ID**: Unique identifier for each user.
- **Item ID**: Unique identifier for each movie.
- **Rating**: Rating given by the user to the movie (1-5 scale).

## Techniques Used
- **TensorFlow & Keras**: Frameworks used to build and train the neural network model.
- **Collaborative Filtering**: A technique used to make predictions based on user-item interactions.
- **Embedding Layers**: Used to represent users and movies in a lower-dimensional space.

## Model Training
The model is trained using the Mean Squared Error (MSE) loss function, optimizing it with the Adam optimizer. The training process involves fitting the model to the training data and validating it on a subset of the data.

## Evaluation Metrics
To evaluate the model's performance, we calculate the Recall@10 metric, which measures the model's ability to recommend relevant items to users.

```python
# Calculate Recall@10 on test data
def calculate_recall(model, test_data, k=10):
    # Implementation...
```

## Recommendations
The recommendation function suggests the top K movies for a specific user based on predicted ratings.

```python
# Recommend movies for a specific user
def recommend_movies(model, user_id, k=10):
    # Implementation...
```

## Conclusion
This movie recommendation system demonstrates the effectiveness of collaborative filtering using neural networks. The model can be further improved with additional features, hyperparameter tuning, and more complex architectures.

## Requirements
- Python 3.x
- TensorFlow
- Pandas
- Scikit-learn
- Matplotlib

## How to Run
1. Clone the repository.
2. Install the required packages.
3. Run the script to train the model and generate recommendations.
