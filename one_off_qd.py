from gspread import Worksheet
from pprint import pprint
from services.gspread import gspread_try_get_all_records, gspread_try_get_service_account_from_file, gspread_try_get_spreadsheet_by_id, gspread_try_get_worksheet_by_id
import traceback

from sheets.one_off.sheet_question_distribution import update_question_distribution_sheet


def main():
    service_account = gspread_try_get_service_account_from_file('secret_sales_meetings_report_service_account.json')
    spreadsheet = gspread_try_get_spreadsheet_by_id(service_account, '19AYpEl2TeqUAAZ-pqGU3c9j6rtzzFbWgRrEjHA_ozYM') # Production spreadsheet

    # Data sheets
    sheet_closed_won_meetings = gspread_try_get_worksheet_by_id(spreadsheet, 748843719)
    sheet_closed_lost_meetings = gspread_try_get_worksheet_by_id(spreadsheet, 1513545925)

    # Report sheets
    sheet_question_distribution = gspread_try_get_worksheet_by_id(spreadsheet, 1963712501)
    meeting_sheets = (sheet_closed_won_meetings, sheet_closed_lost_meetings)
    question_distribution_df = update_question_distribution_sheet(meeting_sheets, sheet_question_distribution)

    return question_distribution_df


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        raise e
