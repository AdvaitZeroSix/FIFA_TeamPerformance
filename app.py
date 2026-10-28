import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="FIFA Team Performance Predictor",
    layout="wide"
)


# Load the model and data once, not on every Streamlit rerun
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_teams():
    return pd.read_csv("data/male_teams.csv")


model = load_model()
teams_df = load_teams()

teams_df = teams_df[
    [
        "team_name",
        "overall",
        "attack",
        "midfield",
        "defence",
        "starting_xi_average_age",
        "whole_team_average_age",
        "international_prestige",
        "domestic_prestige"
    ]
]

teams_df = teams_df.drop_duplicates(subset="team_name")

st.title("FIFA Team Performance Predictor")

st.write(
    "Predict whether a football team belongs to the Weak, Average, Good, or Elite performance tier."
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Team Prediction",
        "EDA Visualizations",
        "Team Comparison",
        "Model Performance"
    ]
)

with tab1:

    st.header("Enter Team Attributes")

    col1, col2 = st.columns(2)

    with col1:

        attack = st.number_input(
            "Attack",
            min_value=0,
            max_value=100,
            value=80
        )

        midfield = st.number_input(
            "Midfield",
            min_value=0,
            max_value=100,
            value=80
        )

        defence = st.number_input(
            "Defence",
            min_value=0,
            max_value=100,
            value=80
        )

        international_prestige = st.number_input(
            "International Prestige",
            min_value=1,
            max_value=10,
            value=5
        )

    with col2:

        starting_age = st.number_input(
            "Starting XI Average Age",
            min_value=15.0,
            max_value=45.0,
            value=26.0
        )

        whole_team_age = st.number_input(
            "Whole Team Average Age",
            min_value=15.0,
            max_value=45.0,
            value=25.0
        )

        domestic_prestige = st.number_input(
            "Domestic Prestige",
            min_value=1,
            max_value=10,
            value=5
        )

    if st.button("Predict Team Performance"):

        input_data = pd.DataFrame({
            "attack": [attack],
            "midfield": [midfield],
            "defence": [defence],
            "starting_xi_average_age": [starting_age],
            "whole_team_average_age": [whole_team_age],
            "international_prestige": [international_prestige],
            "domestic_prestige": [domestic_prestige]
        })

        prediction = model.predict(input_data)

        probabilities = model.predict_proba(input_data)[0]

        classes = model.classes_

        confidence = probabilities.max() * 100

        if prediction[0] == "Elite":
            st.success(f"Predicted Performance Tier: {prediction[0]}")

        elif prediction[0] == "Good":
            st.info(f"Predicted Performance Tier: {prediction[0]}")

        elif prediction[0] == "Average":
            st.warning(f"Predicted Performance Tier: {prediction[0]}")

        else:
            st.error(f"Predicted Performance Tier: {prediction[0]}")

        probability_df = pd.DataFrame({
            "Performance Tier": classes,
            "Probability": probabilities
        })

with tab2:

    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of Team Overall Ratings")
        st.image(
            "images/overallRatings.png",
            use_container_width=True
        )

    with col2:
        st.subheader("Performance Tier Distribution")
        st.image(
            "images/PerformanceTier.png",
            use_container_width=True
        )

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Confusion Matrix")
        st.image(
            "images/ConfusionMatrix.png",
            use_container_width=True
        )
    with col4:
        st.subheader("Attack Rating")
        st.image(
            "images/attackRatings.png",
            use_container_width=True
        )

with tab3:

    st.header("Team Comparison")

    team_names = sorted(
        teams_df["team_name"].unique()
    )

    col1, col2 = st.columns(2)

    with col1:
        team1 = st.selectbox(
            "Select Team",
            team_names
        )

    with col2:
        team2 = st.selectbox(
            "Select Opponent",
            team_names,
            index=1
        )

    team1_data = teams_df[
        teams_df["team_name"] == team1
    ].iloc[0]

    team2_data = teams_df[
        teams_df["team_name"] == team2
    ].iloc[0]

    comparison = pd.DataFrame({
        "Statistic": [
            "Overall",
            "Attack",
            "Midfield",
            "Defence",
            "Starting XI Average Age",
            "Whole Team Average Age",
            "International Prestige",
            "Domestic Prestige"
        ],
        team1: [
            team1_data["overall"],
            team1_data["attack"],
            team1_data["midfield"],
            team1_data["defence"],
            team1_data["starting_xi_average_age"],
            team1_data["whole_team_average_age"],
            team1_data["international_prestige"],
            team1_data["domestic_prestige"]
        ],
        team2: [
            team2_data["overall"],
            team2_data["attack"],
            team2_data["midfield"],
            team2_data["defence"],
            team2_data["starting_xi_average_age"],
            team2_data["whole_team_average_age"],
            team2_data["international_prestige"],
            team2_data["domestic_prestige"]
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )
    st.subheader("Statistical Comparison")

    stats = [
        ("Overall", "overall"),
        ("Attack", "attack"),
        ("Midfield", "midfield"),
        ("Defence", "defence"),
        ("Starting XI Average Age", "starting_xi_average_age"),
        ("Whole Team Average Age", "whole_team_average_age"),
        ("International Prestige", "international_prestige"),
        ("Domestic Prestige", "domestic_prestige")
    ]

    results = []

    for stat_name, column in stats:

        value1 = team1_data[column]
        value2 = team2_data[column]

        if value1 > value2:
            better = team1
        elif value2 > value1:
            better = team2
        else:
            better = "Equal"

        results.append(
            {
                "Statistic": stat_name,
                "Better Team": better
            }
        )

    st.dataframe(
        pd.DataFrame(results),
        use_container_width=True
    )

    st.subheader("Overall Comparison")

    score1 = (
        team1_data["attack"] +
        team1_data["midfield"] +
        team1_data["defence"] +
        team1_data["international_prestige"] +
        team1_data["domestic_prestige"]
    )

    score2 = (
        team2_data["attack"] +
        team2_data["midfield"] +
        team2_data["defence"] +
        team2_data["international_prestige"] +
        team2_data["domestic_prestige"]
    )

    if score1 > score2:
        st.success(f"{team1} has stronger overall team attributes.")
    elif score2 > score1:
        st.success(f"{team2} has stronger overall team attributes.")
    else:
        st.info("Both teams have identical overall attributes.")

with tab4:

    st.header("Model Performance")

    comparison = pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "Decision Tree"
            ],
            "Accuracy": [
                99.2,
                98.7
            ]
        }
    )

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.metric(
        label="Best Model",
        value="Random Forest",
        delta="99.2% Accuracy"
    )

    st.caption(
        "Note: the performance tier is defined by cutting the team's overall "
        "rating into bands (Weak, Average, Good, Elite). Attack, midfield and "
        "defence feed into that rating, so this accuracy shows how well the "
        "model recovers the rating bands, not how well it forecasts results."
    )

    st.subheader("Project Workflow")

    workflow = pd.DataFrame(
        {
            "Step": [
                "Data Cleaning",
                "Exploratory Data Analysis",
                "Feature Engineering",
                "Model Building",
                "Model Evaluation",
                "Deployment"
            ],
            "Description": [
                "Removed duplicate records and handled missing values",
                "Visualized overall ratings and performance tiers",
                "Created the Performance Tier target variable",
                "Trained Random Forest and Decision Tree models",
                "Compared model accuracy and selected the best model",
                "Built an interactive Streamlit application"
            ]
        }
    )

    st.dataframe(
        workflow,
        use_container_width=True
    )