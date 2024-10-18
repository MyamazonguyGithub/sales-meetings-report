import pandas as pd
from decimal import Decimal, getcontext

from algorithms.emails import get_speaker_email_list

def calculate_talk_to_listen_durations(df: pd.DataFrame) -> float:
    """
    Calculate talk duration for each speaker and the total no-talk (silence) duration from a transcript CSV file.

    This function reads a CSV file containing transcript data with columns: "Speaker", "Transcript",
    "start_time", and "end_time". It calculates the total speaking time for each speaker and the total
    duration of silence (no-talk time) between speaking intervals.

    Args:
        filepath (str): The path to the CSV file containing the transcript data.

    Returns:
    dict: A dictionary containing the following keys:
        - total_duration (float): The total duration of the transcript in seconds.
        - speaker_durations (dict): A dictionary mapping speaker names to their total speaking duration in seconds.
        - no_talk_duration (float): The total duration of silence (no-talk time) in the transcript
    """    
    try:
        # Set decimal precision
        getcontext().prec = 10

        # Calculate duration of each utterance
        df['duration'] = df['end_time'] - df['start_time']

        # Sum durations per speaker
        speaker_durations = df.groupby('speaker_name')['duration'].sum().to_dict()

        # Create a list of all speech intervals
        intervals = df[['start_time', 'end_time']].values.tolist()
        intervals.sort()

        # Merge overlapping intervals to find total talk time
        total_talk_time = 0.0
        merged_intervals = []
        start, end = intervals[0]
        for curr_start, curr_end in intervals[1:]:
            if curr_start <= end:
                end = max(end, curr_end)
            else:
                merged_intervals.append((start, end))
                total_talk_time += end - start
                start, end = curr_start, curr_end
        merged_intervals.append((start, end))
        total_talk_time += end - start

        # Calculate total duration and no-talk duration
        total_duration = df['end_time'].max() - df['start_time'].min()
        no_talk_duration = total_duration - total_talk_time

        # Convert Decimal values to float for the final output
        speaker_durations = {name: float(duration_seconds) for name, duration_seconds in speaker_durations.items()}
        no_talk_duration = float(no_talk_duration)
        
        ae_emails = get_speaker_email_list()

        ae_talk_duration = df[df['speaker_email'].isin(ae_emails)]['duration'].sum()
        client_talk_duration = df[~df['speaker_email'].isin(ae_emails)]['duration'].sum()


        return {
            'total_duration': float(total_duration),
            'speaker_durations': speaker_durations,
            'ae_talk_duration': float(ae_talk_duration),
            'client_talk_duration': float(client_talk_duration),
            'no_talk_duration': no_talk_duration
        }
    except Exception as e:
        return {
            'total_duration': 0.0,
            'speaker_durations': {},
            'ae_talk_duration': 0.0,
            'client_talk_duration': 0.0,
            'no_talk_duration': 0.0
        }
