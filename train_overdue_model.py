import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Suponha que df_receivable já esteja carregado

df_receivable = pd.read_excel("inputs/Databases/versao_demo/trades_receivable.xlsx")  
df_receivable['overdue_ratio'] = df_receivable['overdue'] / df_receivable['trades_receivable']
df_receivable['month'] = pd.to_datetime(df_receivable['month_year']).dt.month
df_receivable['year'] = pd.to_datetime(df_receivable['month_year']).dt.year
df_receivable['country_encoded'] = df_receivable['country'].astype('category').cat.codes

features = ['dso', 'sales', 'cei', 'art', 'month', 'year', 'country_encoded']
target = 'overdue_ratio'

X = df_receivable[features]
y = df_receivable[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Salvar o modelo
joblib.dump(model, "overdue_risk_model.joblib")
