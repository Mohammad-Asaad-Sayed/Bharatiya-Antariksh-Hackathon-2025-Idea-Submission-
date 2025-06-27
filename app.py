# === Forest Fire Risk Prediction and Simulation - Streamlit App (Real Data) ===

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Title ---
st.title("Forest Fire Risk Prediction & Spread Simulation")
st.markdown("Predict fire-prone areas using real UCI data and visualize spread simulation")

# --- Load dataset ---
@st.cache_data

def load_data():
    file_path = r'https://github.com/Mohammad-Asaad-Sayed/Bharatiya-Antariksh-Hackathon-2025-Idea-Submission-/blob/main/forestfires.csv'
    df = pd.read_csv(file_path)
    df['fire_risk'] = df['area'].apply(lambda x: 1 if x > 0 else 0)
    return df

data = load_data()

# --- Sidebar Input ---
st.sidebar.header("Input Fire Weather Indices")
FFMC = st.sidebar.slider("FFMC", 18.7, 96.2, 85.0)
DMC = st.sidebar.slider("DMC", 1.1, 291.3, 100.0)
DC = st.sidebar.slider("DC", 7.9, 860.6, 400.0)
ISI = st.sidebar.slider("ISI", 0.0, 56.1, 10.0)
temp = st.sidebar.slider("Temperature (°C)", 2.2, 33.3, 20.0)
RH = st.sidebar.slider("Relative Humidity (%)", 15, 100, 50)
wind = st.sidebar.slider("Wind Speed (km/h)", 0.4, 9.4, 4.0)
rain = st.sidebar.slider("Rain (mm)", 0.0, 6.4, 0.0)

user_input = pd.DataFrame({
    'FFMC': [FFMC], 'DMC': [DMC], 'DC': [DC], 'ISI': [ISI],
    'temp': [temp], 'RH': [RH], 'wind': [wind], 'rain': [rain]
})

st.subheader("User Input Features")
st.write(user_input)

# --- Model Training ---
features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
X = data[features]
y = data['fire_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Prediction ---
pred = model.predict(user_input)[0]
prob = model.predict_proba(user_input)[0][1]

st.subheader("🔥 Fire Risk Prediction")
if pred == 1:
    st.error(f"High Fire Risk Detected ({prob*100:.2f}% probability)")
else:
    st.success(f"Low Fire Risk ({(1-prob)*100:.2f}% probability)")

# --- Fire Spread Simulation ---
st.subheader("🔥 Simple Fire Spread Simulation")

size = 20
grid = np.zeros((size, size))
center = size // 2
grid[center, center] = 1

def spread_fire(grid, steps=3):
    new_grid = grid.copy()
    for _ in range(steps):
        new = new_grid.copy()
        for i in range(1, size - 1):
            for j in range(1, size - 1):
                if new_grid[i, j] == 1:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if np.random.rand() < 0.3:
                                new[i + dx, j + dy] = 1
        new_grid = new
    return new_grid

simulated_grid = spread_fire(grid, steps=5)

fig, ax = plt.subplots()
sns.heatmap(simulated_grid, cmap="YlOrRd", cbar=False, ax=ax)
ax.set_title("Simulated Fire Spread Map")
st.pyplot(fig)

st.markdown("---")
st.caption("Demo App - Forest Fire Modeling using Real Dataset")
