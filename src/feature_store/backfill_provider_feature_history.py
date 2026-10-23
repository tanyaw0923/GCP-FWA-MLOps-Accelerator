import argparse
import calendar
from datetime import date

from dateutil.relativedelta import relativedelta

from src.feature_store.refresh_provider_feature_history import (
    refresh_provider_feature_history,
)


def month_end(value: date) -> date:
    return date(
        value.year,
        value.month,
        calendar.monthrange(
            value.year,
            value.month,
        )[1],
    )


def iter_month_ends(
    start_date: date,
    end_date: date,
):
    current = start_date.replace(day=1)
    final_month = end_date.replace(day=1)

    while current <= final_month:
        snapshot_date = month_end(current)

        if (
            snapshot_date >= start_date
            and snapshot_date <= end_date
        ):
            yield snapshot_date

        current = (
            current
            + relativedelta(months=1)
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Backfill monthly provider feature-history snapshots "
            "for point-in-time model training."
        )
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="First eligible snapshot date (YYYY-MM-DD).",
    )

    parser.add_argument(
        "--end-date",
        required=True,
        help="Last eligible snapshot date (YYYY-MM-DD).",
    )

    args = parser.parse_args()

    start_date = date.fromisoformat(
        args.start_date
    )
    end_date = date.fromisoformat(
        args.end_date
    )

    if end_date < start_date:
        raise ValueError(
            "end-date must be on or after start-date."
        )

    snapshots = list(
        iter_month_ends(
            start_date=start_date,
            end_date=end_date,
        )
    )

    if not snapshots:
        raise ValueError(
            "No month-end snapshots fall inside the "
            "requested date range."
        )

    print("=" * 70)
    print("BACKFILL PROVIDER FEATURE HISTORY")
    print("=" * 70)
    print(
        f"Snapshots to build: {len(snapshots)}"
    )

    for snapshot_date in snapshots:
        print(
            "\nBuilding snapshot: "
            f"{snapshot_date}"
        )

        refresh_provider_feature_history(
            snapshot_date=snapshot_date
        )

    print(
        "\nProvider feature-history backfill complete."
    )


if __name__ == "__main__":
    main()
