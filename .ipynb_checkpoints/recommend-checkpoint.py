import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np

# Load dataset
df = pd.read_csv("netflix_titles.csv")

# Display the first few rows
print("Dataset Preview:")
print(df.head())

# Ensure 'listed_in' is processed into lowercase lists
df['listed_in'] = df['listed_in'].apply(lambda x: [genre.strip().lower() for genre in x.split(',')] if isinstance(x, str) else [])

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["description"].fillna(""))  # Avoid NaN errors

def auto_extract_keywords(user_input):
    """
    Automatically extracts genre-related keywords from user input.
    """
    genre_keywords = []
    genre_terms = ["action", "sci-fi", "comedy", "adventure", "space", "alien", "thriller", "drama", "mystery", "fantasy"]
    
    # Convert user input to lowercase
    user_input = user_input.lower()
    
    # Check for the presence of genre-related keywords in the user input
    for term in genre_terms:
        if term in user_input:
            genre_keywords.append(term)
    
    print("Extracted Genre Keywords:", genre_keywords)  # Debugging line
    return genre_keywords

def recommend(user_input, top_n=5, type_filter=None, min_genre_score=3, recent=False):
    """
    Content-based recommendation with genre prioritization based on user input.

    Parameters:
    - user_input (str): The user's description of desired content.
    - top_n (int): Number of recommendations.
    - type_filter (str): Optional filter for 'Movie' or 'TV Show'.
    - min_genre_score (int): Minimum required genre match score.
    - recent (bool): If True, prioritize newer releases.

    Returns:
    - DataFrame with recommended titles.
    """
    
    # Automatically extract genre keywords from user input
    genre_keywords = auto_extract_keywords(user_input)
    
    # Boosted query: Add space-related or user-specified terms
    boosted_query = (user_input + " " + " ".join(genre_keywords))
    
    # Compute cosine similarity
    input_vec = vectorizer.transform([boosted_query])
    similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()
    
    # Fetch top_n candidates and filter by similarity
    top_indices = similarities.argsort()[-top_n*5:][::-1]  # Fetch more candidates
    recommended_items = df.iloc[top_indices].copy()
    recommended_items = recommended_items[similarities[top_indices] > 0.1]  # Apply similarity threshold

    # Adjust genre filtering
    genre_boosted_keywords = ["action", "sci-fi", "space", "adventure", "thriller", "alien", "drama", "mystery", "comedy", "fantasy"]
    
    def genre_match_score(row_genres):
        """
        Calculate a score for how well the genres of the content match the user input.
        We'll consider partial matches (e.g., "Sci-Fi" for "space").
        """
        return sum(1 for genre in row_genres if any(keyword in genre for keyword in genre_boosted_keywords))
    
    recommended_items['genre_match_score'] = recommended_items['listed_in'].apply(genre_match_score)
    
    # Debugging: Show genre match score distribution
    print("Genre Match Scores Distribution:")
    print(recommended_items[['title', 'genre_match_score']].head())
    
    recommended_items = recommended_items[recommended_items['genre_match_score'] > 0]  # Only keep items that match at least one genre

    # Apply type filter if needed
    if type_filter:
        recommended_items = recommended_items[recommended_items["type"].str.contains(type_filter, case=False, na=False)]

    # Prioritize recent content using exponential regression
    if recent:
        recommended_items['age_weight'] = recommended_items['release_year'].apply(lambda x: np.exp(-0.1 * (2025 - x)))  # Assume 2025 is the current year
    else:
        recommended_items['age_weight'] = 1  # No weight for non-recent content

    # Sort primarily by genre match score and secondarily by release year (with weight)
    sorted_recommendations = recommended_items.sort_values(by=["genre_match_score", "age_weight"], ascending=False)

    # Limit to top_n recommendations
    top_recommendations = sorted_recommendations.head(top_n)

    # Print recommendations in the requested format
    if not top_recommendations.empty:
        for index, row in top_recommendations[["title", "listed_in", "description"]].iterrows():
            print(f"title: {row['title']}")
            print(f"listed_in: {', '.join(row['listed_in'])}")
            print(f"description: {row['description'][:150]}...")  # Truncate description if it's too long
            print("-" * 40)  # Separator between recommendations
    else:
        print("No recommendations found.")

# Example Test: Action-packed space movies with comedy
user_query = "I love thrilling action movies set in space, with a comedic twist."

print("Recommendations:")
recommendations = recommend(user_query, top_n=5, recent=True)
