from pprint import pprint
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def calculate_question_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the question distribution in a transcript.

    This function reads a DataFrame containing transcript data with columns: "Speaker", "Transcript",
    "start_time", and "end_time". It classifies each transcript as a question or statement using a pre-trained
    question-vs-statement classifier model.

    Args:
        df (pd.DataFrame): The DataFrame containing the transcript data.

    Returns:
        pd.DataFrame: The DataFrame containing the transcript data with an additional column "Question" that
            indicates whether the transcript is a question (True) or a statement (False).
    """
    # Create question pipeline
    tokenizer = AutoTokenizer.from_pretrained('shahrukhx01/question-vs-statement-classifier')
    model = AutoModelForSequenceClassification.from_pretrained('shahrukhx01/question-vs-statement-classifier')
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

    # Create a new column in the DataFrame to store the classification results
    df["is_question"] = df["text"].apply(lambda x: False if classifier(x)[0]["label"] == "LABEL_0" else True)

    # # Identify speakers
    df["is_ae"] = df["speaker_email"].apply(lambda x: True if x is not None and 'myamazonguy' in x else False)
    
    # Filter salesperson's utterances
    ae_df = df[(df['is_ae'] & df['is_question'])].copy()
    customer_df = df[(~df['is_ae'] & df['is_question'])].copy()

    # Total Number of Questions
    ae_total_questions = ae_df['is_question'].sum()
    client_total_questions = customer_df['is_question'].sum()

    # Calculate total call duration in seconds
    call_start = df['start_time'].min()
    call_end = df['end_time'].max()
    call_duration_seconds = call_end - call_start
    call_duration_minutes = call_duration_seconds / 60

    # Questions per Minute
    ae_questions_per_minute = ae_total_questions / call_duration_minutes if call_duration_minutes > 0 else 0
    client_questions_per_minute = client_total_questions / call_duration_minutes if call_duration_minutes > 0 else 0

    # First Question Timing
    ae_first_question_time = ae_df[ae_df['is_question'] == True]['start_time'].min()
    if not np.isnan(ae_first_question_time):
        ae_first_question_timing_seconds = ae_first_question_time - call_start

    client_first_question_time = customer_df[customer_df['is_question'] == True]['start_time'].min()
    if not np.isnan(client_first_question_time):
        client_first_question_timing_seconds = client_first_question_time - call_start

    # Number of segments
    num_segments = 10
    segment_duration = call_duration_seconds / num_segments

    # Initialize question counts per segment
    ae_questions_per_segment = [0] * num_segments
    client_questions_per_segment = [0] * num_segments

    # Assign each question to a segment
    for index, row in ae_df[ae_df['is_question'] == True].iterrows():
        time_since_start = row['start_time'] - call_start
        segment_index = int(time_since_start // segment_duration)
        if segment_index >= num_segments:
            segment_index = num_segments - 1  # Handle edge case
        ae_questions_per_segment[segment_index] += 1

    for index, row in customer_df[customer_df['is_question'] == True].iterrows():
        time_since_start = row['start_time'] - call_start
        segment_index = int(time_since_start // segment_duration)
        if segment_index >= num_segments:
            segment_index = num_segments - 1  # Handle edge case
        client_questions_per_segment[segment_index] += 1

    # Calculate probabilities
    ae_total_questions_in_segments = sum(ae_questions_per_segment)
    if ae_total_questions_in_segments > 0:
        ae_probabilities = [q / ae_total_questions_in_segments for q in ae_questions_per_segment]
        # Calculate entropy
        ae_entropy = -sum([p * np.log2(p) for p in ae_probabilities if p > 0])
        # Normalize entropy
        ae_max_entropy = np.log2(num_segments)
        ae_normalized_entropy = ae_entropy / ae_max_entropy if ae_max_entropy > 0 else 0

    client_total_questions_in_segments = sum(client_questions_per_segment)
    if client_total_questions_in_segments > 0:
        client_probabilities = [q / client_total_questions_in_segments for q in client_questions_per_segment]
        # Calculate entropy
        client_entropy = -sum([p * np.log2(p) for p in client_probabilities if p > 0])
        # Normalize entropy
        client_max_entropy = np.log2(num_segments)
        client_normalized_entropy = client_entropy / client_max_entropy if client_max_entropy > 0 else 0

    def get_gini_coefficient(x):
        x = np.array(x, dtype=np.float64)  # Ensure x is of type float
        total = 0
        for i, xi in enumerate(x[:-1], 1):
            total += np.sum(np.abs(xi - x[i:]))
        return total / (len(x)**2 * np.mean(x))

    # Calculate Gini Coefficient
    if ae_total_questions_in_segments > 0:
        ae_gini = get_gini_coefficient(ae_questions_per_segment)

    if client_total_questions_in_segments > 0:
        client_gini = get_gini_coefficient(client_questions_per_segment)

    # Calculate total speaking time for ae and client
    ae_ask_time = ae_df['end_time'].sum() - ae_df['start_time'].sum()
    client_ask_time = customer_df['end_time'].sum() - customer_df['start_time'].sum()

    total_talk_time = ae_ask_time + client_ask_time
    if total_talk_time > 0:
        ae_question_ratio = ae_ask_time / total_talk_time
        client_question_ratio = client_ask_time / total_talk_time

    # Average Time Between Questions (Normalized)
    ae_question_times = ae_df[ae_df['is_question'] == True]['start_time'].tolist()
    if len(ae_question_times) > 1:
        ae_intervals = [t2 - t1 for t1, t2 in zip(ae_question_times[:-1], ae_question_times[1:])]
        ae_average_interval = np.mean(ae_intervals)

    client_question_times = customer_df[customer_df['is_question'] == True]['start_time'].tolist()
    if len(client_question_times) > 1:
        client_intervals = [t2 - t1 for t1, t2 in zip(client_question_times[:-1], client_question_times[1:])]
        client_average_interval = np.mean(client_intervals)

    # Create a summary dictionary
    metrics_summary = {
        'ae_total_questions': ae_total_questions,
        'ae_question_ratio': round(ae_question_ratio, 2) if total_talk_time > 0 else None,
        'ae_questions_per_minute': round(ae_questions_per_minute, 2),
        'ae_first_question_timing_seconds': round(ae_first_question_timing_seconds, 2) if not np.isnan(ae_first_question_time) else None,
        'ae_entropy': ae_normalized_entropy if ae_total_questions_in_segments > 0 else None,
        'ae_gini_coefficient': ae_gini if ae_total_questions_in_segments > 0 else None,
        'ae_ave_time_between_questions_seconds': round(ae_average_interval, 2) if len(ae_question_times) > 1 else None,
        'ae_questions_per_segment': ae_questions_per_segment,
        'client_total_questions': client_total_questions,
        'client_question_ratio': round(client_question_ratio, 2) if total_talk_time > 0 else None,
        'client_questions_per_minute': round(client_questions_per_minute, 2),
        'client_first_question_timing_seconds': round(client_first_question_timing_seconds, 2) if not np.isnan(client_first_question_time) else None,
        'client_entropy': client_normalized_entropy if client_total_questions_in_segments > 0 else None,
        'client_gini_coefficient': client_gini if client_total_questions_in_segments > 0 else None,
        'client_ave_time_between_questions_seconds': round(client_average_interval, 2) if len(client_question_times) > 1 else None,
        'client_questions_per_segment': client_questions_per_segment,
    }

    return metrics_summary
