import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import contractions
from collections import Counter
import matplotlib.pyplot as plt

# Load CSV file
R1 = pd.read_csv("C:\\Users\\User\\Documents\\spyderfile\\reviews_0-250.csv")
R2 = pd.read_csv("C:\\Users\\User\\Documents\\spyderfile\\reviews_250-500.csv")
R3 = pd.read_csv("C:\\Users\\User\\Documents\\spyderfile\\reviews_500-750.csv")
R4 = pd.read_csv("C:\\Users\\User\\Documents\\spyderfile\\reviews_750-1250.csv")
R5 = pd.read_csv("C:\\Users\\User\\Documents\\spyderfile\\reviews_1250-end.csv")

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Text cleaning function
def clean_text(text):
    # Expand contractions
    text = contractions.fix(text)
    
    # Lowercasing
    text = text.lower()
    
    # Removing unwanted characters (handles Ã¢â‚¬â„¢ and similar)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Removing numbers
    text = re.sub(r'\d+', '', text)
    
    # Removing special characters
    text = re.sub(r'\W', ' ', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Joining tokens back into a single string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# Check for missing values in 'review_text' and drop those rows
R1 = R1.dropna(subset=['review_text'])

# Ensure all entries in 'review_text' are strings
R1['review_text'] = R1['review_text'].astype(str)

# Clean the review text
R1['cleaned_review'] = R1['review_text'].apply(clean_text)

# Perform sentiment analysis
R1['sentiment_polarity'] = R1['cleaned_review'].apply(lambda text: TextBlob(text).sentiment.polarity)

# Save the processed data
R1.to_csv('R1_sentiment_reviews.csv', index=False)


################

#a.Word Count
# Load (Cleaned) CSV files
R1 = pd.read_csv("C:\\Users\\User\\Documents\\spyderfile\\R1_sentiment_reviews.csv")
R2 = pd.read_csv("C:\\Users\\User\\Documents\\spyderfile\\R2_sentiment_reviews.csv")
R3 = pd.read_csv("C:\\Users\\User\\Documents\\spyderfile\\R3_sentiment_reviews.csv")
R4 = pd.read_csv("C:\\Users\\User\\Documents\\spyderfile\\R4_sentiment_reviews.csv")
R5 = pd.read_csv("C:\\Users\\User\\Documents\\spyderfile\\R5_sentiment_reviews.csv")

# Combine the DataFrames
R = pd.concat([R1, R2, R3, R4, R5])

# Convert 'cleaned_review' column to string type to handle any non-string entries
R['cleaned_review'] = R['cleaned_review'].astype(str)

# Tokenize all cleaned reviews into individual words
all_words = ' '.join(R['cleaned_review']).split()

# Count the frequency of each word
word_freq = Counter(all_words)

# Get the top 20 most common words
top_20_words = word_freq.most_common(20)

# Print the top 20 words
print("Top 20 Most Frequent Words:")
for word, freq in top_20_words:
    print(f"{word}: {freq}")
    


#b.Sentiment Result
# Define a function to classify sentiment
def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the function to create a new column for sentiment classification
R['sentiment_type'] = R['sentiment_polarity'].apply(classify_sentiment)

# Count the occurrences of each sentiment type
sentiment_counts = R['sentiment_type'].value_counts()


# Plot a pie chart
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['green', 'red', 'blue'])
plt.title('Sentiment Distribution')
plt.show()


#c.WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import numpy as np

# Limit the vocabulary size
vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=10000)  # Use unigrams only

# Fit and transform to create the sparse matrix
X = vectorizer.fit_transform(R['cleaned_review'])

# Sum up the counts for each word (sparse representation)
word_freq = np.asarray(X.sum(axis=0)).flatten()
words = vectorizer.get_feature_names_out()

# Combine the names and frequencies
word_counts = dict(zip(words, word_freq))

# Sort and get the top N words
top_n_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:500]


# Create a word cloud for the top words
all_text_words = ' '.join([word for word, freq in top_n_words])
wordcloud_words = WordCloud(width=800, height=400, background_color='white').generate(all_text_words)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_words, interpolation='bilinear')
plt.axis('off')  # No axes for the word cloud
plt.title('Word Cloud of Top Words')
plt.show()



  