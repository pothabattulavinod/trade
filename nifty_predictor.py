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

print("🤖 Live AI 5-Min Candle Predictor for NIFTY")

# ⏳ Download NIFTY 5-min data
print("⏳ Fetching NIFTY 5-min data...")
nifty = yf.download("^NSEI", interval="5m", period="5d", progress=False)

if nifty.empty:
    print("❌ Data fetch failed. Check market status or API limits.")
    exit()

# 🧠 Feature Engineering
nifty['Return'] = nifty['Close'].pct_change()
nifty['Direction'] = np.where(nifty['Return'].shift(-1) > 0, 1, 0)
nifty['MA5'] = nifty['Close'].rolling(5).mean()
nifty['MA10'] = nifty['Close'].rolling(10).mean()
nifty['Volatility'] = nifty['Close'].rolling(5).std()
nifty.dropna(inplace=True)

# 📊 Features
features = ['Close', 'MA5', 'MA10', 'Volatility']
X = nifty[features]
y = nifty['Direction']

# 🔍 Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# 🧠 Train model
model = RandomForestClassifier(n_estimators=200, max_depth=5)
model.fit(X_train, y_train)

# 📈 Evaluate (last 5 predictions)
last_5_pred = model.predict(X_scaled[-5:])
last_5_true = y.values[-5:]
accuracy = np.mean(last_5_pred == last_5_true) * 100
print(f"✅ Model Backtest Accuracy (last 5 candles): {accuracy:.2f}%")

# 🔮 Prediction
latest = nifty.iloc[[-1]]
latest_time_ist = latest.index[0].astimezone(ZoneInfo("Asia/Kolkata"))
latest_close = float(latest['Close'].iloc[0])
latest_features = scaler.transform(latest[features])
pred = model.predict(latest_features)[0]

# 🎯 Target and SL logic
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

# 🖨️ Print Recommendation
print("\n📈 --- Trade Recommendation ---")
print(f"🕒 Candle Time: {latest_time_ist.strftime('%Y-%m-%d %H:%M:%S')} (IST)")
print(f"💰 Last Close: {latest_close:.2f}")
print(f"📡 Prediction: {direction}")
print(f"🎯 Target: {target:.2f}")
print(f"🛑 Stoploss: {stoploss:.2f}")
print(f"💡 Trade Suggestion: {option} — Buy")
print(f"🕘 Logged at: {now_ist.strftime('%Y-%m-%d %H:%M:%S')} (IST)")

# 💬 WhatsApp message body
whatsapp_message = f"""📈 --- Trade Recommendation ---
🕒 Candle Time: {latest_time_ist.strftime('%Y-%m-%d %H:%M:%S')} (IST)
💰 Last Close: {latest_close:.2f}
📡 Prediction: {direction}
🎯 Target: {target:.2f}
🛑 Stoploss: {stoploss:.2f}
💡 Trade Suggestion: {option} — Buy
🕘 Logged at: {now_ist.strftime('%Y-%m-%d %H:%M:%S')} (IST)
"""

# ✅ Twilio credentials from environment variables (GitHub Secrets)
account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
from_whatsapp = 'whatsapp:+14155238886'

# 📲 Recipients
recipient_numbers = [
     'whatsapp:+917731965666',
        'whatsapp:+918520876451',
        'whatsapp:+919032821222'
]

# 🚀 Send WhatsApp messages
try:
    client = Client(account_sid, auth_token)
    for number in recipient_numbers:
        message = client.messages.create(
            from_=from_whatsapp,
            body=whatsapp_message,
            to=number
        )
        print(f"✅ WhatsApp sent to {number} | SID: {message.sid}")
except Exception as e:
    print(f"❌ WhatsApp message failed: {e}")
