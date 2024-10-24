def get_indices() -> tuple:
    """
    Get the start and end indices for the slice number.

    Returns:
        tuple: The start and end indices.
    """
    import sys

    # Check if the correct number of arguments are provided
    if len(sys.argv) != 2:
        print("Usage: python sheet_metrics_per_transcript.py <slice_number>")
        sys.exit(1)

    slice_number = int(sys.argv[1])

    # Define the slice size and total length
    HOURS_PER_RUN = 6
    SLICE_SIZE = 24 * HOURS_PER_RUN
    TOTAL_LENGTH = 2815

    # Calculate the last start index
    LAST_START_INDEX = (TOTAL_LENGTH - 1) // SLICE_SIZE * SLICE_SIZE

    # Check if the slice number is valid
    if slice_number < 1 or slice_number > (TOTAL_LENGTH + SLICE_SIZE - 1) // SLICE_SIZE:
        raise ValueError(f"Slice number must be between 1 and {(TOTAL_LENGTH + SLICE_SIZE - 1) // SLICE_SIZE}")

    # Calculate start and end indices
    start_index = (slice_number - 1) * SLICE_SIZE
    end_index = min(slice_number * SLICE_SIZE, TOTAL_LENGTH)

    print(f'{start_index} to {end_index}')

    return start_index, end_index