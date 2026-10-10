import math
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    r"D:\MAJOR PROJECT DATA\cert r4.2\r4.2\r4.2\http.csv"
)

OUTPUT_FILE = Path(
    r"D:\MAJOR PROJECT DATA\cert r4.2\r4.2\r4.2\http_sample.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_ROWS = 1_500_000
CHUNK_SIZE = 100_000


# ============================================================
# COUNT ROWS
# ============================================================

def count_rows():

    print("=" * 70)
    print("STEP 1: COUNTING HTTP ROWS")
    print("=" * 70)

    total_rows = 0

    for chunk in pd.read_csv(
        INPUT_FILE,
        usecols=["id"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        total_rows += len(chunk)

        print(
            f"Rows counted so far: {total_rows:,}"
        )

    return total_rows


# ============================================================
# CREATE SYSTEMATIC SAMPLE
# ============================================================

def create_sample(total_rows):

    print("\n" + "=" * 70)
    print("STEP 2: CREATING REPRESENTATIVE SAMPLE")
    print("=" * 70)

    # Calculate spacing between selected rows.
    step = max(
        1,
        math.ceil(
            total_rows / TARGET_ROWS
        )
    )

    estimated_rows = math.ceil(
        total_rows / step
    )

    print(
        f"\nTotal HTTP rows: {total_rows:,}"
    )

    print(
        f"Sampling every {step:,} rows"
    )

    print(
        f"Estimated sample size: "
        f"{estimated_rows:,}"
    )

    print(
        f"\nOutput:\n{OUTPUT_FILE}"
    )

    # Remove old sample if it exists.
    if OUTPUT_FILE.exists():

        print(
            "\nRemoving existing sample..."
        )

        OUTPUT_FILE.unlink()

    first_chunk = True

    current_row = 0
    selected_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1
    ):

        print(
            f"\nProcessing chunk {chunk_number}..."
        )

        # Global row numbers for this chunk.
        start = current_row
        end = current_row + len(chunk)

        # Select rows using their GLOBAL position.
        positions = (
            pd.Series(
                range(start, end)
            )
            .mod(step)
            .eq(0)
            .to_numpy()
        )

        sampled = chunk.loc[positions]

        if not sampled.empty:

            sampled.to_csv(
                OUTPUT_FILE,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )

            selected_rows += len(sampled)

            first_chunk = False

        current_row = end

        print(
            f"Sample rows created: "
            f"{selected_rows:,}"
        )

    print("\n" + "=" * 70)
    print("SAMPLING COMPLETE")
    print("=" * 70)

    print(
        f"\nOriginal rows: {total_rows:,}"
    )

    print(
        f"Sample rows: {selected_rows:,}"
    )

    print(
        f"\nSample saved to:"
    )

    print(OUTPUT_FILE)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"HTTP file not found:\n{INPUT_FILE}"
        )

    total_rows = count_rows()

    create_sample(total_rows)