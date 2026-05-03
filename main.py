import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("movies.csv")

# Convert genre text into numbers
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(df['genre'])

# Similarity calculation
similarity = cosine_similarity(genre_matrix)

# Function to recommend movies
def recommend(movie_name):
    index = df[df['title'] == movie_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended movies:")
    for i in scores[1:4]:
        print(df.iloc[i[0]]['title'])

# User input
movie = input("Enter movie name: ")
recommend(movie)
