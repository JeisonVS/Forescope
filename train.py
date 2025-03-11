import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = pd.read_csv('var_filtradas.csv')
data.info()

data['fecha'] = pd.to_datetime(data['fecha'])

data['dayofweek'] = data['fecha'].dt.dayofweek
data['Mes'] = data['fecha'].dt.month
data['Anio'] = data['fecha'].dt.year
data['quarter'] = data['fecha'].dt.quarter
data['day_of_month'] = data['fecha'].dt.day
data['week_of_year'] = data['fecha'].dt.isocalendar().week
data['day_of_year'] = data['fecha'].dt.day_of_year
data['is_leap'] = data['fecha'].dt.is_leap_year
data['cantidad_dias_mes'] = data['fecha'].dt.daysinmonth
data['inicio_mes'] = data['fecha'].dt.is_month_start
data['fin_mes'] = data['fecha'].dt.is_month_end


X = data[['dayofweek', 'Mes', 'Anio', 'quarter',	'day_of_month',	'week_of_year', 'day_of_year', 'is_leap', 'cantidad_dias_mes', 'inicio_mes',	'fin_mes']]
y = data['cantidad.incidente']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

model = RandomForestRegressor(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir métricas
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2): {r2}')

#print(y_pred)

import pickle
pickle.dump(model, open('modelRF.pkl','wb'))

model_RF = pickle.load(open('modelRF.pkl','rb'))

new_date = pd.to_datetime('2020-01-01')


#new_data = [[new_festividad, new_camapnia, new_cts, new_grati, new_date.dayofweek, new_date.month, new_date.year, new_date.quarter, new_date.day, new_date.weekofyear, new_date.day_of_year, new_date.is_leap_year, new_date.daysinmonth, new_date.is_month_start, new_date.is_month_end ]]
new_data = [[new_date.dayofweek, new_date.month, new_date.year, new_date.quarter, new_date.day, new_date.weekofyear, new_date.day_of_year, new_date.is_leap_year, new_date.daysinmonth, new_date.is_month_start, new_date.is_month_end ]]
predicted_demand = model.predict(new_data)
print(f'Predicted Demand for {new_date} : {predicted_demand[0]}')

# def generate_predictions_with_special_months(model, initial_date):
#     # Crear un DataFrame con 10 días siguientes a la fecha inicial
#     dates = pd.date_range(start=initial_date, periods=10, freq='D')
#     dayofweek = dates.dayofweek
#     dates.month
#
#     data = {'dayofweek': dayofweek, 'Mes': dates.month, 'Anio': dates.year, 'quarter': dates.quarter, 'day_of_month': dates.day, 'week_of_year': dates.isocalendar().week , 'day_of_year': dates.day_of_year, 'is_leap': dates.is_leap_year, 'cantidad_dias_mes': dates.daysinmonth, 'inicio_mes': dates.is_month_start , 'fin_mes': dates.is_month_end}
#     future_df = pd.DataFrame(data)
#     predictions = model.predict(future_df)
#     results = pd.DataFrame({'Date': dates, 'Predicted_Demand': predictions})
#     return results
#
# initial_date = pd.to_datetime('2020-01-01')
# predictions_with_special_months = generate_predictions_with_special_months(model, initial_date)
#
# # Imprimir las predicciones
# print(predictions_with_special_months)