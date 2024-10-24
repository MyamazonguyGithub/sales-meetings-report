from gspread import Worksheet
import pandas as pd
from services.gspread import gspread_try_append_rows, gspread_try_clear_with_ranges, gspread_try_update_range


def update_sheet_with_df(sheet: Worksheet, sheet_df: pd.DataFrame) -> None:
    """
    Update the sheet with the data from the DataFrame.

    Args:
        sheet (gspread.Worksheet): The sheet to be updated.
        sheet_df (pd.DataFrame): The DataFrame with the data.

    Returns:
        None
    """

    # Update the header row
    header_row = sheet_df.columns.tolist()
    gspread_try_update_range(sheet, 'A1', [header_row])

    # Replace None with empty strings
    sheet_df = sheet_df.fillna('')

    # Convert the DataFrame to a list of lists
    sheet_data = sheet_df.values.tolist()

    # Update the sheet
    gspread_try_append_rows(sheet, sheet_data)
