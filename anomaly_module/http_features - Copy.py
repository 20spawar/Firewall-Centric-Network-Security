import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import (
    CHUNK_SIZE,
    CONTENT_COLUMN,
    DATE_COLUMN,
    HTTP_DATA_PATH,
    PC_COLUMN,
    URL_COLUMN,
    USER_COLUMN,
)


# ============================================================
# SUSPICIOUS URL TERMS
# ============================================================

SUSPICIOUS_TERMS = [
    "password",
    "passwd",
    "login",
    "admin",
    "cmd",
    "shell",
    "exec",
    "upload",
    "download",
    "script",
    "select",
    "union",
    "drop",
    "insert",
    "delete",
    "../",
    "..\\",
]


def count_suspicious_terms(url):
    if pd.isna(url):
        return 0

    url = str(url).lower()

    return sum(
        term in url
        for term in SUSPICIOUS_TERMS
    )


def extract_domain(url):
    if pd.isna(url):
        return ""

    url = str(url).strip().lower()

    match = re.match(
        r"^(?:https?://)?([^/:?#]+)",
        url
    )

    if match:
        return match.group(1)

    return ""


# ============================================================
# PROCESS ONE CHUNK
# ============================================================

def process_http_chunk(chunk, user_stats):
    """
    Update global user statistics using one HTTP chunk.
    """

    required_columns = [
        DATE_COLUMN,
        USER_COLUMN,
        PC_COLUMN,
        URL_COLUMN,
        CONTENT_COLUMN,
    ]

    for column in required_columns:
        if column not in chunk.columns:
            raise ValueError(
                f"Required column '{column}' was not found."
            )

    chunk = chunk.copy()

    # --------------------------------------------------------
    # Clean string columns
    # --------------------------------------------------------

    chunk[USER_COLUMN] = (
        chunk[USER_COLUMN]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
    )

    chunk[PC_COLUMN] = (
        chunk[PC_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    chunk[URL_COLUMN] = (
        chunk[URL_COLUMN]
        .fillna("")
        .astype(str)
    )

    chunk[CONTENT_COLUMN] = (
        chunk[CONTENT_COLUMN]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # Parse dates
    # --------------------------------------------------------

    chunk["parsed_date"] = pd.to_datetime(
        chunk[DATE_COLUMN],
        errors="coerce"
    )

    chunk["active_day"] = (
        chunk["parsed_date"]
        .dt.date
        .astype(str)
    )

    chunk["hour"] = (
        chunk["parsed_date"]
        .dt.hour
        .fillna(0)
        .astype(int)
    )

    chunk["day_of_week"] = (
        chunk["parsed_date"]
        .dt.dayofweek
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Time behavior
    # --------------------------------------------------------

    chunk["is_night"] = (
        (chunk["hour"] < 6)
        |
        (chunk["hour"] >= 22)
    ).astype(int)

    chunk["is_weekend"] = (
        chunk["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # URL behavior
    # --------------------------------------------------------

    chunk["url_length"] = (
        chunk[URL_COLUMN].str.len()
    )

    chunk["suspicious_url_count"] = (
        chunk[URL_COLUMN]
        .apply(count_suspicious_terms)
    )

    chunk["has_query"] = (
        chunk[URL_COLUMN]
        .str.contains(
            r"\?",
            regex=True,
            na=False
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Content behavior
    # --------------------------------------------------------

    chunk["content_length"] = (
        chunk[CONTENT_COLUMN].str.len()
    )

    # --------------------------------------------------------
    # Update each user's global statistics
    # --------------------------------------------------------

    for user, group in chunk.groupby(USER_COLUMN):

        if user not in user_stats:

            user_stats[user] = {
                "http_request_count": 0,
                "pcs": set(),
                "active_days": set(),
                "night_request_count": 0,
                "weekend_request_count": 0,
                "url_length_sum": 0,
                "max_url_length": 0,
                "content_length_sum": 0,
                "max_content_length": 0,
                "query_request_count": 0,
                "suspicious_url_count": 0,
            }

        stats = user_stats[user]

        # Total requests
        stats["http_request_count"] += len(group)

        # Unique PCs
        stats["pcs"].update(
            group[PC_COLUMN]
            .loc[group[PC_COLUMN] != ""]
            .unique()
        )

        # Unique active days
        stats["active_days"].update(
            group["active_day"]
            .loc[group["active_day"] != "NaT"]
            .unique()
        )

        # Time behavior
        stats["night_request_count"] += int(
            group["is_night"].sum()
        )

        stats["weekend_request_count"] += int(
            group["is_weekend"].sum()
        )

        # URL statistics
        stats["url_length_sum"] += int(
            group["url_length"].sum()
        )

        stats["max_url_length"] = max(
            stats["max_url_length"],
            int(group["url_length"].max())
        )

        # Content statistics
        stats["content_length_sum"] += int(
            group["content_length"].sum()
        )

        stats["max_content_length"] = max(
            stats["max_content_length"],
            int(group["content_length"].max())
        )

        # Query URLs
        stats["query_request_count"] += int(
            group["has_query"].sum()
        )

        # Suspicious URL terms
        stats["suspicious_url_count"] += int(
            group["suspicious_url_count"].sum()
        )


# ============================================================
# CONVERT GLOBAL STATISTICS TO DATAFRAME
# ============================================================

def build_feature_dataframe(user_stats):

    rows = []

    for user, stats in user_stats.items():

        request_count = stats[
            "http_request_count"
        ]

        if request_count > 0:

            avg_url_length = (
                stats["url_length_sum"]
                / request_count
            )

            avg_content_length = (
                stats["content_length_sum"]
                / request_count
            )

        else:

            avg_url_length = 0

            avg_content_length = 0

        rows.append({

            "user": user,

            "http_request_count":
                request_count,

            "unique_pc_count":
                len(stats["pcs"]),

            "active_days":
                len(stats["active_days"]),

            "night_request_count":
                stats["night_request_count"],

            "weekend_request_count":
                stats["weekend_request_count"],

            "avg_url_length":
                round(avg_url_length, 4),

            "max_url_length":
                stats["max_url_length"],

            "avg_content_length":
                round(avg_content_length, 4),

            "max_content_length":
                stats["max_content_length"],

            "query_request_count":
                stats["query_request_count"],

            "suspicious_url_count":
                stats["suspicious_url_count"],
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN FUNCTION
# ============================================================

def build_user_features(
    input_path=HTTP_DATA_PATH,
    chunk_size=CHUNK_SIZE
):

    input_path = Path(input_path)

    if not input_path.exists():

        raise FileNotFoundError(
            f"HTTP dataset not found:\n{input_path}"
        )

    print("=" * 70)
    print("CERT r4.2 HTTP FEATURE EXTRACTION")
    print("=" * 70)

    print("\nInput file:")
    print(input_path)

    print(
        f"\nChunk size: {chunk_size:,}"
    )

    # Global statistics for all users
    user_stats = {}

    chunk_number = 0
    total_rows = 0

    # --------------------------------------------------------
    # Read huge CSV in chunks
    # --------------------------------------------------------

    for chunk in pd.read_csv(
        input_path,
        chunksize=chunk_size,
        low_memory=False,
    ):

        chunk_number += 1
        total_rows += len(chunk)

        print(
            f"\nProcessing chunk {chunk_number}..."
        )

        print(
            f"Rows in chunk: {len(chunk):,}"
        )

        process_http_chunk(
            chunk,
            user_stats
        )

        print(
            f"Total rows processed: "
            f"{total_rows:,}"
        )

        print(
            f"Unique users so far: "
            f"{len(user_stats):,}"
        )

    # --------------------------------------------------------
    # Final user-level feature table
    # --------------------------------------------------------

    print(
        "\nBuilding final user-level feature table..."
    )

    final_features = build_feature_dataframe(
        user_stats
    )

    print(
        f"\nFinal number of users: "
        f"{len(final_features):,}"
    )

    print(
        f"Final feature columns: "
        f"{len(final_features.columns)}"
    )

    return final_features