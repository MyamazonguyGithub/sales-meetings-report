import numpy as np
import pandas as pd
from algorithms.get_transcript import get_transcript_dfs
from algorithms.question_distribution import calculate_question_distribution
from algorithms.sentiment import set_emotional_reciprocity, set_overall_emotional_intensity_score, set_overall_sentiment_score, set_sentiment_balance_ratio, set_speaker_sentiment_contribution, set_speaker_sentiment_trend, set_speaker_sentiment_variability
from algorithms.talk_to_listen_ratio import calculate_talk_to_listen_durations
from gspread import Worksheet
from services.gspread import (
    gspread_try_get_all_records,
)
from sheets.batched_one_offs.batching import get_indices
from sheets.batched_one_offs.util import update_sheet_with_df
from util.progress import print_progress_by_increment


def update_metrics_per_transcript_sheet(
    meeting_sheets: tuple, sheet_question_distribution: Worksheet
) -> None:
    """
    Update the metrics_per_transcript sheet with the data from the meeting sheets.

    Args:
        meeting_sheets (tuple): The closed-won and closed-lost meeting sheets.
            - sheet_closed_won_meetings (gspread.Worksheet): The closed-won meeting sheet.
            - sheet_closed_lost_meetings (gspread.Worksheet): The closed-lost meeting sheet.
        sheet_question_distribution (gspread.Worksheet): The question distribution sheet.
    """

    all_meetings = get_ids_and_outcomes(meeting_sheets)
    start_idx, end_idx = get_indices()
    all_meetings = all_meetings[start_idx:end_idx]
    sheet_df = pd.DataFrame()

    for idx, meeting in enumerate(all_meetings):
        print_progress_by_increment(all_meetings, idx, 1)
        meeting_df = get_meeting_df(meeting)

        if meeting_df.empty:
            continue

        sheet_df = sheet_df.append(meeting_df, ignore_index=True)

    update_sheet_with_df(sheet_question_distribution, sheet_df)

    return sheet_df


def get_ids_and_outcomes(meeting_sheets: tuple) -> list:
    """
    Get the meeting IDs and outcomes from the meeting sheets.

    Args:
        meeting_sheets (tuple): The closed-won and closed-lost meeting sheets.
            - sheet_closed_won_meetings (gspread.Worksheet): The closed-won meeting sheet.
            - sheet_closed_lost_meetings (gspread.Worksheet): The closed-lost meeting sheet.

    Returns:
        list: The meeting IDs and outcomes.
    """

    # Get the data from the meeting sheets
    sheet_closed_won_meetings, sheet_closed_lost_meetings = meeting_sheets
    closed_won_meetings = gspread_try_get_all_records(sheet_closed_won_meetings)
    closed_won_meetings = [
        {"id": meeting["Transcript ID"], "sales_outcome": "closed_won"}
        for meeting in closed_won_meetings
    ]

    closed_lost_meetings = gspread_try_get_all_records(sheet_closed_lost_meetings)
    closed_lost_meetings = [
        {"id": meeting["Transcript ID"], "sales_outcome": "closed_lost"}
        for meeting in closed_lost_meetings
    ]

    # Combine the data
    all_meetings = closed_won_meetings + closed_lost_meetings

    return all_meetings


def get_meeting_df(meeting) -> pd.DataFrame:
    """
    Get the meeting data.

    Args:
        meeting (dict): The meeting data.
            - id (str): The meeting ID.
            - sales_outcome (str): The sales outcome.

    Returns:
        pd.DataFrame: The meeting data
    """
    transcript_dfs = get_transcript_dfs(meeting["id"])
    meeting_df = transcript_dfs["meeting_df"]
    sentences_df = transcript_dfs["sentences_df"]

    if meeting_df.empty:
        return meeting_df

    meeting_df["sales_outcome"] = meeting["sales_outcome"]

    update_meeting_df_with_talk_metrics(meeting_df, sentences_df)
    update_meeting_df_with_question_metrics(meeting_df, sentences_df)
    update_meeting_df_with_sentiment_metrics(meeting_df, sentences_df)

    return meeting_df


def update_meeting_df_with_talk_metrics(meeting_df, sentences_df):
    """
    Calculate talk durations and talk-listen ratios, and update the meeting DataFrame.

    Args:
        meeting_df (pd.DataFrame): The meeting DataFrame to update.
        sentences_df (pd.DataFrame): The sentences DataFrame for calculations.
    """
    # Calculate durations
    durations = calculate_talk_to_listen_durations(sentences_df)
    meeting_df["total_duration"] = durations["total_duration"]
    meeting_df["ae_talk_duration"] = durations["ae_talk_duration"]
    meeting_df["client_talk_duration"] = durations["client_talk_duration"]
    meeting_df["no_talk_duration"] = durations["no_talk_duration"]

    # Extract scalar values using .iloc[0]
    ae_talk_duration = meeting_df["ae_talk_duration"].iloc[0]
    client_talk_duration = meeting_df["client_talk_duration"].iloc[0]
    no_talk_duration = meeting_df["no_talk_duration"].iloc[0]
    total_duration = meeting_df["total_duration"].iloc[0]

    total_talk_time = ae_talk_duration + client_talk_duration

    # Compute ratios using scalar values
    ae_talk_ratio = ae_talk_duration / total_talk_time if total_talk_time > 0 else 0
    client_talk_ratio = client_talk_duration / total_talk_time if total_talk_time > 0 else 0
    ae_talk_ratio_duration = ae_talk_duration / total_duration if total_duration > 0 else 0
    client_talk_ratio_duration = client_talk_duration / total_duration if total_duration > 0 else 0
    no_talk_ratio_duration = no_talk_duration / total_duration if total_duration > 0 else 0

    # Assign scalar values back to meeting_df
    meeting_df["ae_talk_ratio"] = ae_talk_ratio
    meeting_df["client_talk_ratio"] = client_talk_ratio
    meeting_df["ae_talk_ratio_duration"] = ae_talk_ratio_duration
    meeting_df["client_talk_ratio_duration"] = client_talk_ratio_duration
    meeting_df["no_talk_ratio_duration"] = no_talk_ratio_duration



def update_meeting_df_with_question_metrics(meeting_df, sentences_df):
    """
    Calculate question metrics and update the meeting DataFrame.

    Args:
        meeting_df (pd.DataFrame): The meeting DataFrame to update.
        sentences_df (pd.DataFrame): The sentences DataFrame for calculations.
    """
    questions = calculate_question_distribution(sentences_df)

    # AE (Account Executive) question metrics
    meeting_df["ae_total_questions"] = questions["ae_total_questions"]
    meeting_df["ae_question_ratio"] = questions["ae_question_ratio"]
    meeting_df["ae_questions_per_minute"] = questions["ae_questions_per_minute"]
    meeting_df["ae_first_question_timing_seconds"] = questions[
        "ae_first_question_timing_seconds"
    ]
    meeting_df["ae_entropy"] = questions["ae_entropy"]
    meeting_df["ae_gini_coefficient"] = questions["ae_gini_coefficient"]
    meeting_df["ae_ave_time_between_questions_seconds"] = questions[
        "ae_ave_time_between_questions_seconds"
    ]
    meeting_df["ae_questions_per_segment"] = ", ".join(
        map(str, questions["ae_questions_per_segment"])
    )

    # Client question metrics
    meeting_df["client_total_questions"] = questions["client_total_questions"]
    meeting_df["client_question_ratio"] = questions["client_question_ratio"]
    meeting_df["client_questions_per_minute"] = questions[
        "client_questions_per_minute"
    ]
    meeting_df["client_first_question_timing_seconds"] = questions[
        "client_first_question_timing_seconds"
    ]
    meeting_df["client_entropy"] = questions["client_entropy"]
    meeting_df["client_gini_coefficient"] = questions["client_gini_coefficient"]
    meeting_df["client_ave_time_between_questions_seconds"] = questions[
        "client_ave_time_between_questions_seconds"
    ]
    meeting_df["client_questions_per_segment"] = ", ".join(
        map(str, questions["client_questions_per_segment"])
    )


def update_meeting_df_with_sentiment_metrics(meeting_df, sentences_df):
    """
    Calculate sentiment metrics and update the meeting DataFrame.

    Args:
        meeting_df (pd.DataFrame): The meeting DataFrame to update.
        sentences_df (pd.DataFrame): The sentences DataFrame for calculations.
    """

    meeting_df, sentences_df = set_overall_sentiment_score(meeting_df, sentences_df)
    meeting_df, sentences_df = set_overall_emotional_intensity_score(meeting_df, sentences_df)
    meeting_df, sentences_df = set_sentiment_balance_ratio(meeting_df, sentences_df)
    meeting_df, sentences_df = set_speaker_sentiment_contribution(meeting_df, sentences_df)
    meeting_df, sentences_df = set_speaker_sentiment_variability(meeting_df, sentences_df)
    meeting_df, sentences_df = set_speaker_sentiment_trend(meeting_df, sentences_df)
    meeting_df, sentences_df = set_emotional_reciprocity(meeting_df, sentences_df)
