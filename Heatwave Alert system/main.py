import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from twilio.rest import Client

data = pd.read_csv('historical_temperature_data.csv')

features = ['Data.Precipitation', 'Date.Month', 'Date.Week of', 'Date.Year',
            'Data.Temperature.Min Temp', 'Data.Wind.Speed']
target = 'Data.Temperature.Max Temp'

data['Date.Full'] = pd.to_datetime(data['Date.Full'])
data['Date.Month'] = data['Date.Full'].dt.month
data['Date.Week of'] = data['Date.Full'].dt.isocalendar().week
data['Date.Year'] = data['Date.Full'].dt.year

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")

future_dates = pd.date_range(start=data['Date.Full'].max(), periods=8, freq='D')[1:]
future_data = pd.DataFrame({
    'Date.Full': future_dates,
    'Data.Precipitation': np.random.rand(7),
    'Data.Temperature.Min Temp': np.random.rand(7) * 15 + 10,
    'Data.Wind.Speed': np.random.rand(7) * 10 
})
future_data['Date.Month'] = future_data['Date.Full'].dt.month
future_data['Date.Week of'] = future_data['Date.Full'].dt.isocalendar().week
future_data['Date.Year'] = future_data['Date.Full'].dt.year

future_X = future_data[features]
future_data['predicted_temperature'] = model.predict(future_X)

account_sid = ''
auth_token = ''

def send_whatsapp_message(body):
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        from_='whatsapp:',
        body=body,
        to='whatsapp:'
    )
    print(f"WhatsApp message sent: {message.sid}")

message = "Here are the predicted temperatures for the next 7 days:\n\n"

for index, row in future_data.iterrows():
    message += f"Date: {row['Date.Full'].strftime('%Y-%m-%d')}, Predicted Temperature: {row['predicted_temperature']:.2f}°C\n"

send_whatsapp_message(message)

print("WhatsApp message sent successfully!")
