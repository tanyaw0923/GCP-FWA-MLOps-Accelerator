from typing import NamedTuple

from kfp import dsl


@dsl.component(
    base_image="python:3.11",
)
def calculate_rolling_window_component(
    current_data_month: str,
    training_months: int,
    testing_months: int,
) -> NamedTuple(
    "Outputs",
    [
        ("train_start_date", str),
        ("train_end_date", str),
        ("test_start_date", str),
        ("test_end_date", str),
    ],
):
    import calendar
    from datetime import date
    from typing import NamedTuple

    # ---------------------------------------------------------
    # Move a year/month pair forward or backward.
    # ---------------------------------------------------------
    def add_months(
        year,
        month,
        offset,
    ):
        month_index = (
            year * 12
            + month
            - 1
            + offset
        )

        new_year = (
            month_index // 12
        )

        new_month = (
            month_index % 12
            + 1
        )

        return (
            new_year,
            new_month,
        )

    # ---------------------------------------------------------
    # Return the last day of a month.
    # ---------------------------------------------------------
    def month_end(
        year,
        month,
    ):
        final_day = (
            calendar.monthrange(
                year,
                month,
            )[1]
        )

        return date(
            year,
            month,
            final_day,
        )

    current = date.fromisoformat(
        current_data_month
    )

    # ---------------------------------------------------------
    # TEST window
    #
    # The latest complete month is included in TEST.
    #
    # Example:
    #
    # current_data_month = October
    # testing_months = 3
    #
    # TEST = Aug - Oct
    # ---------------------------------------------------------
    test_end = month_end(
        current.year,
        current.month,
    )

    (
        test_start_year,
        test_start_month,
    ) = add_months(
        current.year,
        current.month,
        -(testing_months - 1),
    )

    test_start = date(
        test_start_year,
        test_start_month,
        1,
    )

    # ---------------------------------------------------------
    # TRAIN ends immediately before TEST begins.
    # ---------------------------------------------------------
    (
        train_end_year,
        train_end_month,
    ) = add_months(
        test_start_year,
        test_start_month,
        -1,
    )

    train_end = month_end(
        train_end_year,
        train_end_month,
    )

    # ---------------------------------------------------------
    # TRAIN contains the configured number of complete months.
    #
    # Example:
    #
    # training_months = 6
    #
    # TRAIN = Feb - Jul
    # ---------------------------------------------------------
    (
        train_start_year,
        train_start_month,
    ) = add_months(
        train_end_year,
        train_end_month,
        -(training_months - 1),
    )

    train_start = date(
        train_start_year,
        train_start_month,
        1,
    )

    print("=" * 60)
    print("Rolling Train / Test Window")
    print("=" * 60)

    print(
        f"Current data month: "
        f"{current_data_month}"
    )

    print(
        f"Training months: "
        f"{training_months}"
    )

    print(
        f"Testing months: "
        f"{testing_months}"
    )

    print(
        f"\nTRAIN: "
        f"{train_start} -> {train_end}"
    )

    print(
        f"TEST:  "
        f"{test_start} -> {test_end}"
    )

    outputs = NamedTuple(
        "Outputs",
        [
            ("train_start_date", str),
            ("train_end_date", str),
            ("test_start_date", str),
            ("test_end_date", str),
        ],
    )

    return outputs(
        train_start.isoformat(),
        train_end.isoformat(),
        test_start.isoformat(),
        test_end.isoformat(),
    )
