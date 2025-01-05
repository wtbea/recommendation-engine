import polars as pl

ratings = pl.read_csv('./data/ratings.csv')
movies = pl.read_csv('./data/movies.csv')

n_ratings = len(ratings) 
n_movies = ratings['movieId'].n_unique()
n_users = ratings['userId'].n_unique()

mean_ratings = ratings.group_by('userId').agg(pl.col('rating').mean())

mean_rating_value = mean_ratings['rating'].mean()
rounded_mean_rating = round(mean_rating_value, 2)

movie_ratings = ratings.join(movies, on='movieId')
five_star_ratings = movie_ratings.filter(pl.col('rating') == 5.0)

five_star_counts = five_star_ratings.group_by('title').agg(pl.count('rating').alias('five_star_count'))

top_movies = five_star_counts.sort('five_star_count', descending=True).head(10)

print(f"Number of ratings: {n_ratings}")
print(f"Number of unique movieId's: {n_movies}")
print(f"Number of unique users: {n_users}")
print(f"Average number of ratings per user: {round(n_ratings/n_users, 2)}")
print(f"Average number of ratings per movie: {round(n_ratings/n_movies, 2)}")
print(f"Mean rating per user: {rounded_mean_rating}.")
print(f"Top 10 rated movies: {top_movies}")