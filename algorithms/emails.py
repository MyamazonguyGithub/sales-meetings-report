my_amazon_guy_speakers_emails = {
    "Shawn Henderson": "shawn.henderson@myamazonguy.com",
    "Carmela Ochea": "carmela.ochea@myamazonguy.com",
    "Kimberly Caranay": "kimberly@myamazonguy.com",
    "Kimberly Anne Caranay": "kimberly@myamazonguy.com",
    "John Aspinall": "john.aspinall@myamazonguy.com",
    "Matt Lopez": "matt.lopez@myamazonguy.com",
    "Dan Pope": "dan.pope@myamazonguy.com",
    "Robert Runyon": "robert.runyon@myamazonguy.com"
}


def get_speaker_email_list():
    """
    Get the speaker emails.

    Returns:
        list: The speaker emails.
    """

    return list(my_amazon_guy_speakers_emails.values())


def get_speaker_email_dict():
    """
    Get the speaker emails from the speaker data.

    Returns:
        dict: The speaker emails.
    """

    return my_amazon_guy_speakers_emails
