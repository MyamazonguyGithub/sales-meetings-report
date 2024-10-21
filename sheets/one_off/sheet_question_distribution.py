from gspread import Worksheet
import pandas as pd

from services.gspread import gspread_try_clear_with_ranges, gspread_try_update_range
from sheets.one_off.sheet_talk_to_listen_ratio import get_ids_and_outcomes, get_meeting_df
from sheets.one_off.util import update_sheet_with_df
from util.progress import print_progress_by_increment


def update_question_distribution_sheet(meeting_sheets: tuple, sheet_question_distribution: Worksheet) -> None:
    """
    Update the question_distribution sheet with the data from the meeting sheets.

    Args:
        meeting_sheets (tuple): The closed-won and closed-lost meeting sheets.
            - sheet_closed_won_meetings (gspread.Worksheet): The closed-won meeting sheet.
            - sheet_closed_lost_meetings (gspread.Worksheet): The closed-lost meeting sheet.
        sheet_question_distribution (gspread.Worksheet): The question_distribution sheet.
    """

    all_meetings = get_ids_and_outcomes(meeting_sheets)
    sheet_df = pd.DataFrame()

    for idx, meeting in enumerate(all_meetings):
        print_progress_by_increment(all_meetings, idx, 1)
        meeting_df = get_meeting_df(meeting)

        if meeting_df.empty:
            continue

        sheet_df = sheet_df.append(meeting_df, ignore_index=True)

    update_sheet_with_df(sheet_question_distribution, sheet_df)

    return sheet_df
