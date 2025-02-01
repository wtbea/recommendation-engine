# Recommendation Project

This project is focused on building a recommendation system using movie ratings data. It utilizes the scikit-learn library for machine learning and data analysis.

## Prerequisites

Before running the code, make sure you have the following libraries installed:

- scikit-learn
- polars
- numpy

## Code Overview

The code performs the following steps:

1. Reads the movie ratings and movie data from CSV files.
2. Calculates various statistics, such as the number of ratings, unique movies, and unique users.
3. Computes the mean rating per user and the average number of ratings per user and movie.
4. Creates a sparse matrix representation of the ratings data.
5. Finds similar movies based on a given movie ID using cosine similarity.
6. Prints the similar movies and the number of unique movies in the dataset.
7. Generates binary columns for each genre and calculates the cosine similarity matrix for the genres.
