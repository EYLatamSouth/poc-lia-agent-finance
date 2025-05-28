import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



def predict_overdue_risk(df_receivable: pd.DataFrame, increase_only: bool = True) -> str:
    """
    Predicts future overdue risk based on the ratio of overdue amounts to total accounts receivable.

    This function trains a regression model (Random Forest) using historical receivables data
    to forecast the overdue ratio for the next month, for each country.

    Parameters:
    -----------
    df_receivable : pd.DataFrame
        A DataFrame containing historical receivables data with the following columns:
        - 'month_year', 'country', 'trades_receivable', 'overdue', 'dso', 'sales', 'cei', 'art'.

    increase_only : bool, optional (default=True)
        If True, returns only countries where the predicted overdue ratio increases.
        If False, returns all countries with current and predicted values.

    Returns:
    --------
    str
        A formatted string with the overdue risk forecast for the next month, highlighting countries
        with increased risk, including current, predicted values, and percentage change.
    """

    df = df_receivable.copy()
    df['overdue_ratio'] = df['overdue'] / df['trades_receivable']
    df['month'] = pd.to_datetime(df['month_year']).dt.month
    df['year'] = pd.to_datetime(df['month_year']).dt.year
    df['country_encoded'] = df['country'].astype('category').cat.codes

    features = ['dso', 'sales', 'cei', 'art', 'month', 'year', 'country_encoded']
    target = 'overdue_ratio'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    latest_month = pd.to_datetime(df['month_year'].max())
    next_month = latest_month + pd.DateOffset(months=1)

    result = f"⚠️ *Overdue Risk Forecast* for {next_month.strftime('%B/%Y')}:\n\n"

    for country in df['country'].unique():
        last_row = df[df['country'] == country].sort_values('month_year').iloc[-1].copy()
        last_ratio = last_row['overdue_ratio']

        last_row['month'] = next_month.month
        last_row['year'] = next_month.year
        X_pred = pd.DataFrame([last_row[features]])
        predicted_ratio = model.predict(X_pred)[0]

        delta = predicted_ratio - last_ratio
        if increase_only and delta <= 0:
            continue

        emoji = "🔺" if delta > 0 else "✅"
        result += f"{emoji} {country}: from {last_ratio:.2%} to {predicted_ratio:.2%} ({delta:+.2%})\n"

    return result.strip()


def forecast_liquidity_risk(df_working_capital: pd.DataFrame, threshold: float = 0.0) -> str:
    """
    Forecasts liquidity risk based on historical working capital by country.

    The function trains a regression model to predict working capital for the next month.
    Countries with predictions below a specified threshold are flagged as being at liquidity risk.

    Parameters:
    -----------
    df_working_capital : pd.DataFrame
        A DataFrame with historical working capital data per country, including:
        - 'month_year', 'country', 'working_capital', 'due_interval'.

    threshold : float, optional (default=0.0)
        Minimum expected working capital. Countries with forecasts below this value
        will be flagged as at liquidity risk.

    Returns:
    --------
    str
        A text report listing countries with potential liquidity risk in the next month,
        including actual, predicted values and their difference.
    """

    df = df_working_capital.copy()
    df['month'] = pd.to_datetime(df['month_year']).dt.month
    df['year'] = pd.to_datetime(df['month_year']).dt.year
    df['country_encoded'] = df['country'].astype('category').cat.codes
    df['due_interval_encoded'] = df['due_interval'].astype('category').cat.codes

    features = ['month', 'year', 'country_encoded', 'due_interval_encoded']
    target = 'working_capital'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    latest_month = pd.to_datetime(df['month_year'].max())
    next_month = latest_month + pd.DateOffset(months=1)

    result = f"🔍 *Liquidity Risk Forecast* for {next_month.strftime('%B/%Y')} (threshold = {threshold}):\n\n"

    for country in df['country'].unique():
        last_row = df[df['country'] == country].sort_values('month_year').iloc[-1].copy()
        last_value = last_row['working_capital']

        last_row['month'] = next_month.month
        last_row['year'] = next_month.year
        X_pred = pd.DataFrame([last_row[features]])
        predicted_value = model.predict(X_pred)[0]

        delta = predicted_value - last_value

        if predicted_value < threshold:
            result += (
                f"⚠️ {country}: {last_value:,.2f} → {predicted_value:,.2f} "
                f"({delta:+,.2f}) → liquidity risk\n"
            )

    return result.strip()