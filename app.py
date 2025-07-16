import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import random

# --- Field Explanations ---
field_explanations = {
    "Issue": "Severity rating (1–5). 1 = mild; 5 = critical. Higher scores lower predicted performance.",
    "Interventions": "Support volume (1–5). More support actions reflect higher need and reduce predicted score.",
    "Actions taken": "Response intensity (1–5): 1 = timeout; 5 = external authorities involved. Higher = serious concern.",
    "Best Results": "Success rating of past interventions (1–5). High scores increase predicted performance.",
    "Grade Score": "Academic score (%) before prediction. <30 = high concern; 45–60 = average; 65+ = strong.",
    "School routine": "Engagement with school. Strong routines improve scores.",
    "Home routine": "Stability at home. Good structure supports academic success.",
    "Eating Habits": "Nutrition and regularity. Poor habits may reduce focus and performance.",
    "Aces": "Exposure to trauma or stress. High ACEs = emotional strain and lower prediction.",
    "Social Family": "Family relationship score. Strong support improves engagement.",
    "Social School": "Peer and classroom interaction. Higher scores indicate good social fit.",
    "Outside school acts": "Optional extracurricular activity data. May enrich clustering.",
    "pass scores": "Threshold benchmark for passing. Used to flag students at risk.",
    "predicted score": "AI-generated forecast based on behavior inputs and trained model."
}

# --- Cluster Descriptions ---
cluster_explanations = {
    0: "Cluster 0 = students with higher behavioral need—more interventions, inconsistent routines, elevated ACEs. Support should be structured and restorative.",
    1: "Cluster 1 = students with strong or improving engagement. They benefit from motivational tools and personalized learning strategies."
}

# --- Strategy Links ---
strategy_links = {
    0: [
        "[Challenging Behavior Strategies](https://www.teacherstrategies.org/what-are-some-strategies-for-dealing-with-challenging-student-behaviors/)",
        "[Low-Level Disruption Tips](https://www.behavioursmart.co.uk/post/10-strategies-to-manage-low-level-disruption)",
        "[Support for Distracted Students](https://master-xuan.com/how-to-help-distracted-students/)"
    ],
    1: [
        "[Engagement Strategies](https://www.prodigygame.com/main-en/blog/student-engagement-strategies)",
        "[Motivation Techniques](https://www.teacherstrategies.org/student-engagement-and-motivation-strategies/)",
        "[Positive Psychology in Learning](https://positivepsychology.com/student-engagement/)"
    ]
}

# --- Load Data ---
df = pd.read_excel("student_tracker.xlsx", engine="openpyxl")

behavior_cols = [
    "Issue", "Interventions", "Actions taken", "Best Results", "Grade Score",
    "School routine", "Home routine", "Eating Habits", "Aces",
    "Social Family", "Social School"
]

df = df.dropna(subset=behavior_cols)

# --- Clustering ---
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[behavior_cols])
kmeans = KMeans(n_clusters=2, random_state=42)
df["Behavior Cluster"] = kmeans.fit_predict(scaled_data)

# --- Prediction ---
if "predicted score" in df.columns and df["predicted score"].notna().sum() > 0:
    train_df = df[df["predicted score"].notna()]
    model = LinearRegression()
    model.fit(train_df[behavior_cols], train_df["predicted score"])
    df["score_prediction"] = model.predict(df[behavior_cols])
else:
    df["score_prediction"] = scaled_data.mean(axis=1) * 20 + 50

# --- UI ---
st.title("🎓 Student Behavior Tracker + Forecast Dashboard")

st.subheader("📋 Student Summary")
st.write(f"✅ Loaded {len(df)} student profiles")
st.dataframe(df[["student name", "Behavior Cluster", "score_prediction"]])

# --- Select Student ---
st.subheader("🎯 Select a Student")
selected_name = st.selectbox("Choose a student:", df["student name"])
student_row = df[df["student name"] == selected_name].iloc[0]
cluster = int(student_row["Behavior Cluster"])
score = round(float(student_row["score_prediction"]), 2)
original = student_row["Grade Score"]

# --- Profile Display ---
st.markdown("#### 🧠 Behavior Profile")
st.dataframe(student_row[behavior_cols])
st.markdown(f"**Original Grade Score:** {original}%")
st.markdown(f"**Predicted Academic Score:** {score}%")
st.markdown(f"**Behavior Cluster:** {cluster}")
st.markdown(f"**Cluster Description:** {cluster_explanations[cluster]}")

# --- Bar Chart Comparison ---
st.subheader("📊 Score Comparison")
st.bar_chart(pd.DataFrame({
    "Original Grade": [original],
    "Predicted Score": [score]
}))

# --- Chatbot Section ---
st.subheader("💬 Behavior Chatbot")
st.markdown("_Ask about fields (e.g. 'Issue', 'help Aces'), or type 'strategy' for support suggestions._")
user_input = st.chat_input("Ask about scores, clusters, strategies, or glossary fields")

if user_input:
    st.chat_message("user").markdown(user_input)

    def get_bot_reply(text, cluster):
        text = text.lower()

        for field in field_explanations:
            if field.lower() in text:
                return f"**{field}**: {field_explanations[field]}"

        if "strategy" in text or "help" in text:
            links = strategy_links.get(cluster, [])
            return "🔧 Here are strategies based on this student's cluster:\n\n" + "\n\n".join(links)

        if "score" in text:
            return f"{selected_name}'s original score was {original}%, predicted score is {score}%, based on behavioral inputs."

        if "cluster" in text:
            return f"**Cluster {cluster}**: {cluster_explanations.get(cluster)}"

        return random.choice([
            "Try asking about a specific field like 'Eating Habits' or 'Aces'",
            "Need strategies? Just type 'strategy'",
            "I can explain how behaviors impact prediction—just name a field!"
        ])

    st.chat_message("assistant").markdown(get_bot_reply(user_input, cluster))

# --- Year-over-Year Forecasting ---
st.subheader("📅 Grade Score Forecasting (Year-by-Year)")
st.markdown("Enter scores from the past 5 years (0 = unknown), and we’ll predict year 6 performance.")

past_scores = [st.slider(f"Year {i+1}", 0, 100, value=60) for i in range(5)]
known_years = [i+1 for i, val in enumerate(past_scores) if val > 0]
known_vals = [val for val in past_scores if val > 0]

if len(known_vals) >= 2:
    trend = np.polyfit(known_years, known_vals, 1)
    forecast_score = round(trend[0]*6 + trend[1], 2)

    st.markdown(f"**📈 Predicted Year 6 Grade Score:** {forecast_score}%")
    st.line_chart(pd.DataFrame({
        "Year": known_years + [6],
        "Score": known_vals + [forecast_score]
    }))
else:
    st.warning("Please enter at least two known scores to generate a forecast.")
    