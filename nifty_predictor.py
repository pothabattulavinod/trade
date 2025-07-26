import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from zoneinfo import ZoneInfo
from twilio.rest import Client
import warnings

warnings.filterwarnings("ignore")

print("🤖 Live AI 5-Min Candle Predictor for NIFTY, BANKNIFTY & SENSEX")

# 🧠 Function to run prediction for an index
def predict_index(ticker, name):
    print(f"\n📊 Running for: {name} ({ticker})")
    data = yf.download(ticker, interval="5m", period="5d", progress=False)

    if data.empty:
        print(f"❌ Data fetch failed for {name}")
        return None

    data['Return'] = data['Close'].pct_change()
    data['Direction'] = np.where(data['Return'].shift(-1) > 0, 1, 0)
    data['MA5'] = data['Close'].rolling(5).mean()
    data['MA10'] = data['Close'].rolling(10).mean()
    data['Volatility'] = data['Close'].rolling(5).std()
    data.dropna(inplace=True)

    features = ['Close', 'MA5', 'MA10', 'Volatility']
    X = data[features]
    y = data['Direction']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=200, max_depth=5)
    model.fit(X_train, y_train)

    last_5_pred = model.predict(X_scaled[-5:])
    last_5_true = y.values[-5:]
    accuracy = np.mean(last_5_pred == last_5_true) * 100
    print(f"✅ {name} Accuracy (last 5 candles): {accuracy:.2f}%")

    latest = data.iloc[[-1]]
    latest_time_ist = latest.index[0].astimezone(ZoneInfo("Asia/Kolkata"))
    latest_close = float(latest['Close'].iloc[0])
    latest_features = scaler.transform(latest[features])
    pred = model.predict(latest_features)[0]

    target_pts = 25
    sl_pts = 15

    if pred == 1:
        direction = "🔼 UP"
        target = latest_close + target_pts
        stoploss = latest_close - sl_pts
        option = f"{int(round(latest_close / 50.0) * 50)}CE"
    else:
        direction = "🔽 DOWN"
        target = latest_close - target_pts
        stoploss = latest_close + sl_pts
        option = f"{int(round(latest_close / 50.0) * 50)}PE"

    now_ist = datetime.datetime.now(ZoneInfo("Asia/Kolkata"))

    print(f"🕒 Candle Time: {latest_time_ist.strftime('%Y-%m-%d %H:%M:%S')} (IST)")
    print(f"💰 Last Close: {latest_close:.2f}")
    print(f"📡 Prediction: {direction}")
    print(f"🎯 Target: {target:.2f}")
    print(f"🛑 Stoploss: {stoploss:.2f}")
    print(f"💡 Suggestion: {option} — Buy")
    print(f"🕘 Time: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")

    # Compose message
    msg = f"""📈 --- {name} Trade Recommendation ---
🕒 Candle Time: {latest_time_ist.strftime('%Y-%m-%d %H:%M:%S')} (IST)
💰 Last Close: {latest_close:.2f}
📡 Prediction: {direction}
🎯 Target: {target:.2f}
🛑 Stoploss: {stoploss:.2f}
💡 Trade Suggestion: {option} — Buy
🕘 Logged at: {now_ist.strftime('%Y-%m-%d %H:%M:%S')} (IST)"""
    
    return msg


# ✅ Twilio credentials from GitHub Actions Secrets
account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
from_whatsapp = 'whatsapp:+14155238886'

# 📲 Recipients
recipient_numbers = [
    'whatsapp:+917731965666',
    'whatsapp:+918520876451',
    'whatsapp:+919032821222'
]

# 🧠 Run predictions for all indices
messages = []
indices = {
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "BANK NIFTY",
    "^BSESN": "SENSEX"
}

for ticker, name in indices.items():
    msg = predict_index(ticker, name)
    if msg:
        messages.append(msg)

# 📨 Combine all predictions into one message
combined_message = "\n\n".join(messages)

# 🚀 Send single combined WhatsApp message to each recipient
try:
    client = Client(account_sid, auth_token)
    for number in recipient_numbers:
        message = client.messages.create(
            from_=from_whatsapp,
            body=combined_message,
            to=number
        )
        print(f"✅ WhatsApp sent to {number} | SID: {message.sid}")
except Exception as e:
    print(f"❌ WhatsApp message failed: {e}")
