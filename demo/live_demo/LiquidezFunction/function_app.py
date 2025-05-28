import azure.functions as func
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime
import json
import logging

app = func.FunctionApp()

def predict_overdue_risk(df_receivable: pd.DataFrame, country: str) -> str:
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

    if country not in df['country'].unique():
        return f"Country '{country}' not found in data."

    last_row = df[df['country'] == country].sort_values('month_year').iloc[-1].copy()
    last_ratio = last_row['overdue_ratio']

    last_row['month'] = next_month.month
    last_row['year'] = next_month.year
    X_pred = pd.DataFrame([last_row[features]])
    predicted_ratio = model.predict(X_pred)[0]

    delta = predicted_ratio - last_ratio
    emoji = "🔺" if delta > 0 else "✅"

    result = {
        "country": country,
        "last_overdue_ratio": f"{last_ratio:.2%}",
        "predicted_overdue_ratio": f"{predicted_ratio:.2%}",
        "delta": f"{delta:+.2%}",
        "next_month": next_month.strftime('%B/%Y'),
        "status": "increased" if delta > 0 else "stable/decreased",
        "emoji": emoji
    }

    return result

@app.route(route="previsaoRisco", auth_level=func.AuthLevel.Anonymous)
def previsaoRisco(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
        country = req_body.get('country')
    except ValueError:
        return func.HttpResponse(
            "Invalid request body. Expected JSON with 'country'.",
            status_code=400
        )

    if not country:
        return func.HttpResponse(
            "Missing 'country' in request body.",
            status_code=400
        )

    # Simulando dataframe com dados fictícios
    data = {
        'month_year': ['2024-03-01', '2024-04-01', '2024-05-01'] * 2,
        'country': ['Brazil', 'Brazil', 'Brazil', 'Chile', 'Chile', 'Chile'],
        'trades_receivable': [100000, 110000, 105000, 95000, 97000, 98000],
        'overdue': [5000, 5200, 5300, 4000, 4100, 4200],
        'dso': [45, 46, 44, 50, 49, 48],
        'sales': [200000, 210000, 205000, 195000, 197000, 198000],
        'cei': [0.85, 0.86, 0.84, 0.80, 0.82, 0.83],
        'art': [1.5, 1.6, 1.55, 1.4, 1.45, 1.5]
    }

    df_receivable = pd.DataFrame(data)

    result = predict_overdue_risk(df_receivable, country)

    return func.HttpResponse(
        json.dumps(result, indent=4),
        status_code=200,
        mimetype="application/json"
    )
