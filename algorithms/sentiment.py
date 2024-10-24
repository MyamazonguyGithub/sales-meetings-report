from scipy.stats import linregress, pearsonr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging as transformers_logging, pipeline
import numpy as np
import pandas as pd
import torch.nn.functional as F

# Suppress only the specific warning from transformers
transformers_logging.set_verbosity_error()

# MARK: Sentiment Analysis
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

def calculate_sentiment_classification_scores(sentence: str) -> dict:
    """
    Calculate the sentiment classification scores of a sentence.

    Args:
        sentence (str): The sentence to calculate the sentiment classification for.

    Returns:
        dict: The sentiment classification scores of the sentence.
    """
    # Get probabilities for each label
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    labels = model.config.id2label

    # Map probabilities to labels
    result = {labels[i]: prob.item() for i, prob in enumerate(probs[0])}

    # Ensure all expected keys are present
    for key in ['negative', 'neutral', 'positive']:
        if key not in result:
            result[key] = 0.0
    return result


def set_overall_sentiment_score(meeting_df: pd.DataFrame, sentences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the sentiment score of a conversation based on the sentiment of the sentences in the conversation.

    Args:
        sentences_df (pd.DataFrame): The DataFrame containing the sentences to calculate the sentiment score for.

    Returns:
        pd.DataFrame: The DataFrame containing the sentiment scores of the sentences.
    """
    # Calculate the sentiment classification scores of each sentence and expand the dictionary into separate columns
    sentiment_scores_df = sentences_df['text'].apply(calculate_sentiment_classification_scores).apply(pd.Series)
    sentiment_scores_df = sentiment_scores_df.add_prefix('sentiment_')
    for col in sentiment_scores_df.columns:
        sentences_df[col] = sentiment_scores_df[col]
    sentences_df['sentiment_score'] = -1 * sentences_df['sentiment_negative'] + 0 * sentences_df['sentiment_neutral'] + 1 * sentences_df['sentiment_positive']

    sentences_df['duration'] = sentences_df['end_time'] - sentences_df['start_time']
    sentences_df['cumulative_duration'] = sentences_df['duration'].cumsum()

    sentences_df['weighted_sentiment_score'] = sentences_df['sentiment_score'] * sentences_df['duration']
    overall_sentiment_score = sentences_df['weighted_sentiment_score'].sum() / sentences_df['duration'].sum() if sentences_df['duration'].sum() != 0 else 0

    meeting_df['sentiment_score'] = overall_sentiment_score

    return meeting_df, sentences_df


# MARK: Emotional Analysis
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
excluded_emotions = ['neutral']
emotions = [emotion for emotion in classifier.model.config.id2label.values() if emotion not in excluded_emotions]


def calculate_emotional_scores(sentence: str) -> float:
	"""
	Calculate the emotional scores of a conversation based on the emotional scores of the sentences in the conversation.

	Args:
		sentence (str): The sentence to calculate the emotional scores for.

	Returns:
		float: The emotional scores of the conversation.
	"""

	classifications = classifier(sentence)[0]
	classifications = {
		classification['label']: classification['score']
		for classification in classifications
	}

	return classifications


def calculate_emotional_intensity_score(sentence: str) -> float:
	"""
	Calculate the emotional intensity score of a conversation based on the emotional intensity of the sentences in the conversation.

	Args:
		sentence (str): The sentence to calculate the emotional intensity score for.

	Returns:
		float: The emotional intensity score of the conversation.
	"""

	classifications = calculate_emotional_scores(sentence)

	return 1 - classifications['neutral']


def set_overall_emotional_intensity_score(meeting_df: pd.DataFrame, sentences_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Calculate the emotional intensity score of a conversation based on the emotional intensity of the sentences in the conversation.

	Args:
		sentences_df (pd.DataFrame): The DataFrame containing the sentences to calculate the emotional intensity score for.

	Returns:
		pd.DataFrame: The DataFrame containing the emotional intensity scores of the sentences.
	"""

	sentences_df['emotional_intensity_score'] = sentences_df['text'].apply(calculate_emotional_intensity_score)
	emotional_scores = sentences_df['text'].apply(calculate_emotional_scores).apply(pd.Series)
	for emotion in emotions:
		sentences_df[emotion] = emotional_scores[emotion]

	sentences_df['duration'] = sentences_df['end_time'] - sentences_df['start_time']

	# Calculate the weighted emotional intensity score and weighted emotional scores
	sentences_df['weighted_emotional_intensity_score'] = sentences_df['emotional_intensity_score'] * sentences_df['duration']
	for emotion in emotions:
		sentences_df[f'weighted_{emotion}'] = sentences_df[emotion] * sentences_df['duration']
	
	overall_emotional_intensity_score = sentences_df['weighted_emotional_intensity_score'].sum() / sentences_df['duration'].sum() if sentences_df['duration'].sum() != 0 else 0
	weighted_emotional_scores = {}
	for emotion in emotions:
		weighted_emotional_scores[emotion] = sentences_df[f'weighted_{emotion}'].sum() / sentences_df['duration'].sum() if sentences_df['duration'].sum() != 0 else 0

	meeting_df['emotional_intensity_score'] = overall_emotional_intensity_score
	for emotion in emotions:
		meeting_df[emotion] = weighted_emotional_scores[emotion]

	return meeting_df, sentences_df


# MARK: Sentiment Balance Analysis
def set_sentiment_balance_ratio(meeting_df: pd.DataFrame, sentences_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Calculate the sentiment balance ratio of a conversation based on the sentiment of the sentences in the conversation.

	Args:
		sentences_df (pd.DataFrame): The DataFrame containing the sentences to calculate the sentiment balance ratio for.

	Returns:
		pd.DataFrame: The DataFrame containing the sentiment balance ratio of the sentences.
	"""

	# If sentiment_score do not exist, calculate it
	if 'sentiment_score' not in meeting_df.columns:
		meeting_df, sentences_df = set_overall_sentiment_score(meeting_df, sentences_df)

	# Classify the sentiment of each sentence
	sentences_df['sentiment'] = np.where(sentences_df['sentiment_score'] > 0, 'positive', np.where(sentences_df['sentiment_score'] < 0, 'negative', 'neutral'))

	# Calculate the sentiment balance ratio using log BSR
	sentiment_balance_ratio = np.log(((sentences_df['sentiment'] == 'positive').sum() + 0.000_000_1) / ((sentences_df['sentiment'] == 'negative').sum() + 0.000_000_1))

	meeting_df['sentiment_balance_ratio'] = sentiment_balance_ratio

	return meeting_df, sentences_df



# MARK: Speaker Sentiment Contribution
def set_speaker_sentiment_contribution(meeting_df: pd.DataFrame, sentences_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Calculate the sentiment contribution of each speaker in a conversation based on the sentiment of the sentences spoken by each speaker.

	Args:
		sentences_df (pd.DataFrame): The DataFrame containing the sentences to calculate the sentiment contribution for.

	Returns:
		pd.DataFrame: The DataFrame containing the sentiment contribution of each speaker.
	"""

	# If sentiment_score do not exist, calculate it
	if 'sentiment_score' not in meeting_df.columns:
		meeting_df, sentences_df = set_overall_sentiment_score(meeting_df, sentences_df)

	# Calculate the sentiment contribution of each speaker
	sum_sentiment_scores = sentences_df.groupby('is_account_executive')['sentiment_score'].sum()
	duration_speaker_talks = sentences_df.groupby('is_account_executive')['duration'].sum()

	sentiment_contribution = (sum_sentiment_scores / duration_speaker_talks)

	meeting_df['ae_sentiment'] = sentiment_contribution[True] if True in sentiment_contribution else 0
	meeting_df['client_sentiment'] = sentiment_contribution[False] if False in sentiment_contribution else 0

	return meeting_df, sentences_df


# MARK: Sentiment Variability
def set_speaker_sentiment_variability(meeting_df: pd.DataFrame, sentences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the sentiment variability (weighted standard deviation) of each speaker in a conversation
    based on the sentiment of the sentences spoken by each speaker.

    Args:
        meeting_df (pd.DataFrame): The DataFrame containing meeting-level data.
        sentences_df (pd.DataFrame): The DataFrame containing sentence-level data.

    Returns:
        pd.DataFrame: The updated meeting_df containing the sentiment variability of each speaker.
    """

    # Ensure 'sentiment_score' exists
    if 'sentiment_score' not in sentences_df.columns:
        meeting_df, sentences_df = set_overall_sentiment_score(meeting_df, sentences_df)

    # Define a function to calculate weighted standard deviation
    def weighted_std(values, weights):
        """
        Return the weighted standard deviation.
        values, weights -- Numpy arrays with the same shape.
        """
        average = np.average(values, weights=weights)
        variance = np.average((values - average)**2, weights=weights)
        return np.sqrt(variance)

    variability = {}
    grouped = sentences_df.groupby('is_account_executive')

	# Get sentiment scores and durations
    for is_ae, group in grouped:
        sentiments = group['sentiment_score']
        durations = group['duration']

        if len(sentiments) > 1:
            std = weighted_std(sentiments, durations)
        else:
            std = 0

        variability[is_ae] = std

    # Update meeting_df with the sentiment variability
    meeting_df['ae_sentiment_variability'] = variability.get(True, 0)
    meeting_df['client_sentiment_variability'] = variability.get(False, 0)

    return meeting_df, sentences_df


# MARK: Sentiment Trend
def set_speaker_sentiment_trend(meeting_df: pd.DataFrame, sentences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the sentiment trend (slope over time) of each speaker in a conversation
    based on the sentiment of the sentences spoken by each speaker.

    Args:
        meeting_df (pd.DataFrame): The DataFrame containing meeting-level data.
        sentences_df (pd.DataFrame): The DataFrame containing sentence-level data.

    Returns:
        pd.DataFrame: The updated meeting_df containing the sentiment trend of each speaker.
    """

    # Ensure 'sentiment_score' exists
    if 'sentiment_score' not in sentences_df.columns:
        # Assuming set_overall_sentiment_score function exists and calculates sentiment scores
        meeting_df, sentences_df = set_overall_sentiment_score(meeting_df, sentences_df)

    # Ensure 'utterance_index' exists to represent the sequence of sentences
    if 'utterance_index' not in sentences_df.columns:
        # Assign a sequence number to each sentence in the order they appear
        sentences_df = sentences_df.reset_index(drop=True)
        sentences_df['utterance_index'] = sentences_df.index + 1  # Starts from 1

    # Initialize dictionaries to store sentiment trends
    sentiment_trend = {}

    # Group by 'is_account_executive' to separate speakers
    grouped = sentences_df.groupby('is_account_executive')

    for is_ae, group in grouped:
        # Get sentiment scores and utterance indices
        sentiments = group['sentiment_score']
        indices = group['utterance_index']

        # Check if there are at least two points to calculate a trend
        if len(sentiments) > 1:
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = linregress(indices, sentiments)
        else:
            slope = 0  # If only one sentence, trend is zero

        # Store the sentiment trend
        sentiment_trend[is_ae] = slope

    # Update meeting_df with the sentiment trends
    meeting_df['ae_sentiment_trend'] = sentiment_trend.get(True, 0)
    meeting_df['client_sentiment_trend'] = sentiment_trend.get(False, 0)

    return meeting_df, sentences_df



# MARK: Emotional Reciprocity
def set_emotional_reciprocity(meeting_df: pd.DataFrame, sentences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Emotional Reciprocity between the sales representative and the buyer
    by computing the correlation between their sentiment scores over time.

    Args:
        meeting_df (pd.DataFrame): The DataFrame containing meeting-level data.
        sentences_df (pd.DataFrame): The DataFrame containing sentence-level data.

    Returns:
        pd.DataFrame: The updated meeting_df containing the Emotional Reciprocity metric.
    """

    # Ensure 'sentiment_score' exists
    if 'sentiment_score' not in sentences_df.columns:
        # Assuming set_overall_sentiment_score function exists and calculates sentiment scores
        meeting_df, sentences_df = set_overall_sentiment_score(meeting_df, sentences_df)

    # Ensure 'utterance_index' exists to represent the sequence of sentences
    if 'utterance_index' not in sentences_df.columns:
        # Assign a sequence number to each sentence in the order they appear
        sentences_df = sentences_df.reset_index(drop=True)
        sentences_df['utterance_index'] = sentences_df.index + 1  # Starts from 1

    # Separate sentiments by speaker
    ae_df = sentences_df[sentences_df['is_account_executive'] == True][['utterance_index', 'sentiment_score']]
    client_df = sentences_df[sentences_df['is_account_executive'] == False][['utterance_index', 'sentiment_score']]

    # Merge the two DataFrames on 'utterance_index' using forward and backward filling
    merged_df = pd.merge_asof(
        client_df.sort_values('utterance_index'),
        ae_df.sort_values('utterance_index'),
        on='utterance_index',
        direction='nearest',
        suffixes=('_client', '_ae')
    )

    # Drop rows with missing values (if any)
    merged_df = merged_df.dropna(subset=['sentiment_score_ae', 'sentiment_score_client'])

    # Check if there are at least two data points to calculate correlation
    if len(merged_df) >= 2:
        # Calculate Pearson correlation coefficient
        corr_coef, p_value = pearsonr(merged_df['sentiment_score_ae'], merged_df['sentiment_score_client'])
        emotional_reciprocity = corr_coef
    else:
        emotional_reciprocity = 0  # Not enough data to calculate correlation

    # Update meeting_df with Emotional Reciprocity
    meeting_df['emotional_reciprocity'] = emotional_reciprocity

    return meeting_df, sentences_df

