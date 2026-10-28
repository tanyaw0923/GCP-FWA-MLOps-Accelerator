from src.pipeline.components.calculate_rolling_window_component import (
    calculate_rolling_window_component,
)


def calculate(current_data_month, training_months=6, testing_months=3):
    return calculate_rolling_window_component.python_func(
        current_data_month=current_data_month,
        training_months=training_months,
        testing_months=testing_months,
    )


def test_standard_six_train_three_test_window():
    result = calculate("2026-10-01")

    assert result.train_start_date == "2026-02-01"
    assert result.train_end_date == "2026-07-31"
    assert result.test_start_date == "2026-08-01"
    assert result.test_end_date == "2026-10-31"


def test_window_crosses_year_boundary():
    result = calculate(
        "2026-02-01",
        training_months=6,
        testing_months=3,
    )

    assert result.train_start_date == "2025-06-01"
    assert result.train_end_date == "2025-11-30"
    assert result.test_start_date == "2025-12-01"
    assert result.test_end_date == "2026-02-28"


def test_leap_year_february_end_date():
    result = calculate(
        "2024-02-01",
        training_months=1,
        testing_months=1,
    )

    assert result.train_start_date == "2024-01-01"
    assert result.train_end_date == "2024-01-31"
    assert result.test_start_date == "2024-02-01"
    assert result.test_end_date == "2024-02-29"
