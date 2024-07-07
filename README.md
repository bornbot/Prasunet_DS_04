# Sentiment Analysis and Correlation with Entities

## Overview

This project aims to analyze and visualize sentiment patterns in social media data to understand public opinion and attitudes towards specific topics or brands. The analysis includes:
1. Sentiment Distribution Analysis
2. Sentiment Trends Over Time
3. Hashtag Sentiment Distribution
4. Correlation Analysis between Entities and Sentiment

## Datasets

The project uses two datasets:

1. `twitter_training.csv`
2. `twitter_validation.csv`

Both datasets contain tweets with the following columns:
- `Tweet ID`: Identifier for the tweet.
- `Entity`: The specific topic or brand the tweet is about.
- `Sentiment`: The sentiment label (e.g., Positive, Negative, Neutral, Irrelevant).
- `Tweet content`: The text content of the tweet.

## Files

- `DS_04.ipynb`: Jupyter notebook containing the code for sentiment analysis and visualization.
- `twitter_training.csv`: Training dataset for sentiment analysis.
- `twitter_validation.csv`: Validation dataset for sentiment analysis.

## Installation

To run the code, you'll need to have Python and the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn
```

## Usage
1. Load the Data:
Load the provided datasets into pandas DataFrames.
```python
import pandas as pd

training_data = pd.read_csv('twitter_training.csv')
validation_data = pd.read_csv('twitter_validation.csv')
```

2. Sentiment Distribution Analysis:
Visualize the sentiment distribution across different entities using a bar plot.
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot sentiment distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=training_data, x='Entity', hue='Sentiment', palette='viridis')
plt.title('Sentiment Distribution Across Different Entities')
plt.xticks(rotation=90)
plt.show()
```

3. Sentiment Trends Over Time:
Analyze sentiment trends over time (requires timestamp data, here demonstrated with mock data).
``` python
import numpy as np

# Generate mock timestamp data
np.random.seed(0)
training_data['timestamp'] = pd.date_range(start='2021-01-01', periods=len(training_data), freq='H')

# Group by date and sentiment to count occurrences
sentiment_trends = training_data.groupby([training_data['timestamp'].dt.date, 'Sentiment']).size().unstack().fillna(0)

# Plot sentiment trends over time
plt.figure(figsize=(12, 6))
sentiment_trends.plot(kind='line', marker='o')
plt.title('Sentiment Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.legend(title='Sentiment')
plt.xticks(rotation=45)
plt.show()
```

4. Hashtag Sentiment Distribution:
Extract hashtags and visualize their sentiment distribution.
```python
import re

# Function to extract hashtags from tweet content
def extract_hashtags(text):
    if isinstance(text, str):
        hashtags = re.findall(r'#\w+', text)
        return [hashtag.lower() for hashtag in hashtags]
    return []

# Apply the function to extract hashtags from the tweet content
training_data['hashtags'] = training_data['Tweet content'].apply(extract_hashtags)

# Flatten the list of hashtags and sentiments
hashtag_sentiment_list = [(hashtag, sentiment) for hashtags, sentiment in zip(training_data['hashtags'], training_data['Sentiment']) for hashtag in hashtags]

# Create a DataFrame from the list
hashtag_sentiment_df = pd.DataFrame(hashtag_sentiment_list, columns=['Hashtag', 'Sentiment'])

# Group by hashtag and sentiment to count occurrences
hashtag_sentiment_counts = hashtag_sentiment_df.groupby(['Hashtag', 'Sentiment']).size().unstack(fill_value=0)

# Get the top 20 hashtags by total count
top_hashtags = hashtag_sentiment_counts.sum(axis=1).sort_values(ascending=False).head(20).index
top_hashtag_sentiment_counts = hashtag_sentiment_counts.loc[top_hashtags]

# Plot the sentiment distribution for top hashtags
plt.figure(figsize=(14, 8))
top_hashtag_sentiment_counts.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Sentiment Distribution for Top 20 Hashtags')
plt.xlabel('Hashtags')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.show()
``` 

5. Correlation Analysis:
Examine the relationship between different entities and sentiment.
```python
# Create a list of unique entities
entities = training_data['Entity'].unique()

# Create a binary matrix for entities
entity_matrix = pd.DataFrame(0, index=training_data.index, columns=entities)

# Mark presence of entities in each tweet
for entity in entities:
    entity_matrix[entity] = training_data['Entity'] == entity

# Map sentiment labels to numerical values
sentiment_score_mapping = {
    'Positive': 1,
    'Negative': -1,
    'Neutral': 0,
    'Irrelevant': 0
}
training_data['sentiment_score'] = training_data['Sentiment'].map(sentiment_score_mapping)

# Compute correlation between entity presence and sentiment scores
correlation_matrix = entity_matrix.corrwith(training_data['sentiment_score'])

# Convert the correlation matrix to a DataFrame for better readability
correlation_df = correlation_matrix.reset_index()
correlation_df.columns = ['Entity', 'Correlation']

# Plot the correlation heatmap
plt.figure(figsize=(16, 8))
sns.heatmap(correlation_df.set_index('Entity').T, annot=True, cmap='coolwarm', center=0, fmt=".2f",
            annot_kws={"size": 12}, cbar_kws={'label': 'Correlation'}, linewidths=0.5, linecolor='lightgrey')
plt.title('Correlation Between Entities and Sentiment Scores', fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.show()
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
The sentiment analysis and visualization techniques were inspired by various open-source projects and academic research.
Special thanks to the contributors of the libraries used in this project.
