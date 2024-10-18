from algorithms.get_transcript import get_transcript_dfs
from algorithms.talk_to_listen_ratio import calculate_talk_to_listen_durations
from gspread import Worksheet
from pprint import pprint
from services.gspread import gspread_try_clear_with_ranges, gspread_try_get_all_records, gspread_try_update_range
from util.progress import print_progress_by_increment
import pandas as pd



def update_talk_to_listen_ratio_sheet(meeting_sheets: Worksheet, sheet_talk_to_listen_ratio: Worksheet) -> None:
    """
    Update the talk-to-listen ratio sheet with the data from the meeting sheets.

    Args:
        meeting_sheets (tuple): The closed-won and closed-lost meeting sheets.
            - sheet_closed_won_meetings (gspread.Worksheet): The closed-won meeting sheet.
            - sheet_closed_lost_meetings (gspread.Worksheet): The closed-lost meeting sheet.
        sheet_talk_to_listen_ratio (gspread.Worksheet): The talk-to-listen ratio sheet.

    Returns:
        None
    """
    all_meetings = get_ids_and_outcomes(meeting_sheets)
    sheet_df = pd.DataFrame()

    for idx, meeting in enumerate(all_meetings):
        print_progress_by_increment(all_meetings, idx, 1)
        meeting_df = get_meeting_df(meeting)

        if meeting_df.empty:
            continue

        sheet_df = sheet_df.append(meeting_df, ignore_index=True)

    update_talk_to_listen_ratio_sheet_with_df(sheet_talk_to_listen_ratio, sheet_df)

    return sheet_df

    

def get_ids_and_outcomes(meeting_sheets: Worksheet) -> dict:
    """
    Get the meeting IDs and outcomes from the meeting sheets.

    Args:
        meeting_sheets (tuple): The closed-won and closed-lost meeting sheets.
            - sheet_closed_won_meetings (gspread.Worksheet): The closed-won meeting sheet.
            - sheet_closed_lost_meetings (gspread.Worksheet): The closed-lost meeting sheet.

    Returns:
        dict: The meeting IDs and outcomes.
    """

    # Get the data from the meeting sheets
    sheet_closed_won_meetings, sheet_closed_lost_meetings = meeting_sheets
    closed_won_meetings = gspread_try_get_all_records(sheet_closed_won_meetings)
    closed_won_meetings = [
        {
            'id': meeting['Transcript ID'],
            'sales_outcome': 'closed_won',
        }
        for meeting in closed_won_meetings
    ]

    closed_lost_meetings = gspread_try_get_all_records(sheet_closed_lost_meetings)
    closed_lost_meetings = [
        {
            'id': meeting['Transcript ID'],
            'sales_outcome': 'closed_lost',
        }
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

    transcript_dfs = get_transcript_dfs(meeting['id'])
    sales_outcome = meeting['sales_outcome']
    meeting_df = transcript_dfs['meeting_df']
    sentences_df = transcript_dfs['sentences_df']

    durations = calculate_talk_to_listen_durations(sentences_df)
    meeting_df['sales_outcome'] = sales_outcome
    meeting_df['total_duration'] = durations['total_duration']
    meeting_df['ae_talk_duration'] = durations['ae_talk_duration']
    meeting_df['client_talk_duration'] = durations['client_talk_duration']
    meeting_df['no_talk_duration'] = durations['no_talk_duration']

    total_talk_time = durations['ae_talk_duration'] + durations['client_talk_duration']
    meeting_df['ae_talk_ratio'] = durations['ae_talk_duration'] / total_talk_time if total_talk_time > 0 else 0
    meeting_df['client_talk_ratio'] = durations['client_talk_duration'] / total_talk_time if total_talk_time > 0 else 0

    total_duration = durations['total_duration']
    meeting_df['ae_talk_ratio_duration'] = durations['ae_talk_duration'] / total_duration if total_duration > 0 else 0
    meeting_df['client_talk_ratio_duration'] = durations['client_talk_duration'] / total_duration if total_duration > 0 else 0
    meeting_df['no_talk_ratio_duration'] = durations['no_talk_duration'] / total_duration if total_duration > 0 else 0

    return meeting_df


def update_talk_to_listen_ratio_sheet_with_df(sheet_talk_to_listen_ratio: Worksheet, sheet_df: pd.DataFrame) -> None:
    """
    Update the talk-to-listen ratio sheet with the data from the DataFrame.

    Args:
        sheet_talk_to_listen_ratio (gspread.Worksheet): The talk-to-listen ratio sheet.
        sheet_df (pd.DataFrame): The DataFrame with the data.

    Returns:
        None
    """

    # Clear the sheet
    gspread_try_clear_with_ranges(sheet_talk_to_listen_ratio, ['A3:ZZ'])

    # Print the data as 2D list
    sheet_data = sheet_df.values.tolist()
    gspread_try_update_range(sheet_talk_to_listen_ratio, 'A3', sheet_data)
