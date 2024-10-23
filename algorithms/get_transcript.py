from services import fireflies
import pandas as pd
from util.dates import convert_iso_datetime_to_datetime, convert_to_eastern_time, get_date_formats


def get_transcript_dfs(transcript_id: str) -> dict:
    """
    Pulls transcript and AI data from Fireflies API and creates dataframes for meeting and sentences data.

    Args:
        transcript_id (str): Fireflies transcript ID.

    Returns:
        dict: A dictionary containing two dataframes:
            - 'meeting_df': DataFrame with meeting details.
            - 'sentences_df': DataFrame with sentences and their AI filters, including speaker emails and is_account_executive.
    """
    response_data = fireflies.get_transcript_and_ai_data(transcript_id)

    # Get meeting data
    transcript_data = response_data['data']['transcript']
    attendees = transcript_data['meeting_attendees'] or []
    attendee_emails = [attendee['email'] for attendee in attendees if attendee['email']]
    attendee_email_list = ', '.join(attendee_emails)

    date_string = transcript_data['dateString']
    transcript_datetime = convert_iso_datetime_to_datetime(date_string)
    transcript_eastern_datetime = convert_to_eastern_time(transcript_datetime)
    date_formats = get_date_formats(transcript_eastern_datetime)
    pretty_date = date_formats['datetime_slashed']

    meeting_data = {
        'id': transcript_data['id'],
        'title': transcript_data['title'],
        'ae_name': None,
        'ae_email': None,
        'sales_outcome': None,
        'date': pretty_date,
        'meeting_attendees': attendee_email_list,
        'host_email': transcript_data['host_email'],
        'transcript_url': transcript_data['transcript_url'],
        'video_url': transcript_data['video_url'],
        'audio_url': transcript_data['audio_url'],
    }

    meeting_df = pd.DataFrame([meeting_data])

    # Get sentences data
    sentences_data = transcript_data['sentences']

    # Mapping of known speakers to emails
    my_amazon_guy_speakers_emails = {
        "Shawn Henderson": "shawn.henderson@myamazonguy.com",
        "Carmela Ochea": "carmela.ochea@myamazonguy.com",
        "Kimberly Caranay": "kimberly@myamazonguy.com",
        "Kimberly Anne Caranay": "kimberly@myamazonguy.com",
        "John Aspinall": "john.aspinall@myamazonguy.com",
        "Matt Lopez": "matt.lopez@myamazonguy.com",
        "Matthew Lopez": "matt.lopez@myamazonguy.com",
        "Dan Pope": "dan.pope@myamazonguy.com",
        "Robert Runyon": "robert.runyon@myamazonguy.com"
    }

    # Build mapping from normalized speaker names to emails
    name_to_email = {}
    for attendee in attendees:
        email = attendee.get('email')
        if email:
            if attendee.get('displayName'):
                normalized_name = attendee['displayName'].strip().lower()
            else:
                # Extract name from email if displayName is None
                name_part = email.split('@')[0]
                name_part = name_part.replace('.', ' ').replace('_', ' ').lower()
                normalized_name = name_part
            name_to_email[normalized_name] = email

    # Add known speakers to the mapping
    for name, email in my_amazon_guy_speakers_emails.items():
        normalized_name = name.strip().lower()
        name_to_email[normalized_name] = email

    # For each sentence, add speaker_email and is_account_executive based on speaker_name
    for sentence in sentences_data if sentences_data else []:
        if sentence.get('speaker_name') == None:
            sentence['speaker_name'] = "Unknown"
        
        sentence.update(sentence.pop('ai_filters', {}))
        speaker_name = sentence.get('speaker_name', '').strip().lower()
        email = name_to_email.get(speaker_name)
        if not email:
            # Attempt to match speaker_name with names in name_to_email using partial matching
            for name_key in name_to_email:
                if speaker_name in name_key or name_key in speaker_name:
                    email = name_to_email[name_key]
                    break
        sentence['speaker_email'] = email
        # Determine if speaker is an account executive
        sentence['is_account_executive'] = email in my_amazon_guy_speakers_emails.values() if email else False

    sentences_df = pd.DataFrame(sentences_data)

    # Add AE details
    ae_email = next((email for email in attendee_emails if email in my_amazon_guy_speakers_emails.values()), None)
    meeting_df['ae_email'] = ae_email
    meeting_df['ae_name'] = next((name for name, email in my_amazon_guy_speakers_emails.items() if email == ae_email), None)


    return {
        'meeting_df': meeting_df,
        'sentences_df': sentences_df
    }