# Necessary Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Streamlit settings for wider layout
st.set_page_config(layout="wide")


# Data Loading
@st.cache_data
def load_data():
    data = pd.read_csv("/Users/teamplat/Desktop/Dashboard/FORDASHBOARD.csv", index_col=0)
    data2 = pd.read_csv("/Users/teamplat/Desktop/Dashboard/main_data_features_preprocessed.csv", index_col=0)

    # Your preprocessing steps go here
    # Convert the columns to datetime
    data['Contract end date'] = pd.to_datetime(data['Contract end date'])
    data['Contract start date'] = pd.to_datetime(data['Contract start date'])
    data['Contract creation date'] = pd.to_datetime(data['Contract creation date'])
    data['contract_duration'] = (data['Contract end date'] - data['Contract start date']).dt.days
    data['scores_numeric'] = data['scores'].astype('int')
    data['YearMonth'] = data['Contract creation date'].dt.to_period('M')
    data['ContractYear'] = pd.to_datetime(data['Contract creation date']).dt.year

    return data, data2


data, data2 = load_data()

# Streamlit Sidebar for Section Selection
section = st.sidebar.selectbox(
    'Choose a Section',
    ('Dashboard Home', "Exploratory data analysis", "Score Analysis", "SHAP value Analysis")
)

# Page Title
st.title("Data Analysis Dashboard")

# Colors and Theme
sns.set_theme(style="whitegrid")
PALETTE = "viridis"

if section == 'Dashboard Home':
    st.write("""
    ## Welcome to the Data Analysis Dashboard. 
    Use the sidebar to navigate to different sections.
    """)

elif section == 'Exploratory data analysis':
    plt.figure(figsize=(6, 3))
    st.subheader('Distribution of Car Countries')
    sns.countplot(data=data, y='car_country', order=data['car_country'].value_counts().index, palette=PALETTE)
    st.pyplot(plt, use_container_width=False)


    # st.subheader('Distribution of Vehicle Types')
    # plt.figure(figsize=(6, 3))
    # sns.countplot(data=data, y='Vehicle Type', order=data['Vehicle Type'].value_counts().index, palette=PALETTE)
    # st.pyplot(plt)

    st.subheader('Distribution of Vehicle Types')
    plt.figure(figsize=(6, 3))
    sns.histplot(data=data, x='age', bins=30, kde=True)
    st.pyplot(plt, use_container_width=False)


    # st.subheader('Distribution of Vehicle Types')
    # plt.figure(figsize=(10, 6))
    # sns.countplot(data=data, y='Vehicle Type', order=data['Vehicle Type'].value_counts().index)
    # st.pyplot(plt)

    plt.figure(figsize=(6, 3))
    data['Contract creation date'].value_counts().resample('M').sum().plot()
    plt.title('Contracts Created Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Contracts Created')
    st.pyplot(plt,use_container_width=False)

    plt.figure(figsize=(7, 3))
    sns.lineplot(data=data, x='ContractYear', y='scores_numeric', ci=None, marker="o")
    plt.title('Average Score over the Years')
    plt.xlabel('Year of Contract Creation')
    plt.ylabel('Average Score')
    st.pyplot(plt)


elif section == "Score Analysis":
    st.subheader('Distribution of Vehicle Types')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, y='Vehicle Type', x='scores',
                order=data.groupby('Vehicle Type')['scores'].mean().sort_values(ascending=False).index)

    st.pyplot(plt)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, y='car_country', x='scores', ci=None,
                order=data.groupby('car_country')['scores'].mean().sort_values(ascending=False).index)
    st.pyplot(plt)

    plt.figure(figsize=(14, 7))
    data.groupby('YearMonth')['scores'].mean().plot()
    plt.title('Trend of Average Scores Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Scores')
    st.pyplot(plt)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x='scores', y='Engine power', ci=None, palette="viridis")
    plt.title('Mean Engine Power by Score')
    plt.xlabel('Score')
    plt.ylabel('Average Engine Power')
    st.pyplot(plt)


    plt.figure(figsize=(15, 6))
    sns.boxplot(data=data, x='scores', y='daily_probs', palette="viridis")
    plt.title('Distribution of Daily Probabilities by Score')
    plt.xlabel('Score')
    plt.ylabel('Daily Probabilities')
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, y='Vehicle prod year', x='scores')
    plt.title('Vehicle Production Year vs. Scores')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    age_cols = ['age_less_22', 'age_23_27', 'age_28_35', 'age_36_49', 'age_50_65', 'age_more_65']

    # Compute the mean scores for each age group
    age_distributions = [data2[data2[col] == 1]['scores'].value_counts(normalize=True).sort_index() for col in age_cols]
    plt.figure(figsize=(15, 10))
    for age_dist, age_label in zip(age_distributions, age_cols):
        plt.plot(age_dist.index, age_dist.values, label=age_label)

    plt.legend()
    plt.title("Score Distributions by Age Group")
    plt.xlabel("Scores")
    plt.ylabel("Proportion")
    plt.xticks(list(range(16)))
    st.pyplot(plt)

    country_cols = ['Country_Armenia', 'Country_Russian Federation', 'Country_Iran_Islamic Republic of',
                    'Country_other', 'Country_Georgia']

    # Compute the mean scores for each country
    country_distributions = [data2[data2[col] == 1]['scores'].value_counts(normalize=True).sort_index() for col in
                             country_cols]

    plt.figure(figsize=(15, 10))
    for country_dist, country_label in zip(country_distributions, country_cols):
        plt.plot(country_dist.index, country_dist.values, label=country_label)

    plt.legend()
    plt.title("Score Distributions by Country")
    plt.xlabel("Scores")
    plt.ylabel("Proportion")
    plt.xticks(list(range(16)))
    st.pyplot(plt)

    top_n_models = data['Model'].value_counts().nlargest(5).index
    filtered_data = data[data['Model'].isin(top_n_models)]

    plt.figure(figsize=(14, 7))
    sns.countplot(data=filtered_data, x='Model', hue='scores')
    plt.title('Count of Scores by Top 5 Car Models')
    plt.xticks(rotation=45)
    plt.legend(title='Scores', loc='upper right')
    st.pyplot(plt)

    plt.figure(figsize=(14, 7))
    sns.boxplot(x='scores', y='no_accident_days', data=data)
    plt.title('Distribution of Days Since Last Accident by Score')
    plt.xlabel('Scores')
    plt.ylabel('Days since Last Accident')
    st.pyplot(plt)

    plt.figure(figsize=(14, 7))
    data.groupby("scores").mean('Patm_vtar_qanak')['Patm_vtar_qanak'].plot(kind='bar')
    plt.title('Distribution of Mean Patm_vtar_qanakl by Score')
    plt.xlabel('Scores')
    plt.ylabel('Days since Last Accident')
    st.pyplot(plt)

    plt.figure(figsize=(14, 7))
    data.groupby("scores").mean('Patm_gumarayin_paym_tev')['Patm_gumarayin_paym_tev'].plot(kind='bar')
    plt.title('Distribution of Mean Patm_gumarayin_paym_tev by Score')
    plt.xlabel('Scores')
    plt.ylabel('Days since Last Accident')
    st.pyplot(plt)

    plt.figure(figsize=(14, 7))
    pd.DataFrame(data.groupby("scores").sum("contract_days_passed_duration")["contract_days_passed_duration"] /
                 data.groupby("scores").sum("accident_number")["accident_number"]).plot(kind="bar")
    plt.title('Distribution contract_days_passed_duration/accident_number by Scores')
    plt.xlabel('Scores')
    plt.ylabel('Days since Last Accident')
    st.pyplot(plt)


elif section == 'SHAP value Analysis':
    st.subheader('SHAP value Analysis')

    min_date = data['Contract creation date'].min().date()  # Convert to Python's native date
    max_date = data['Contract creation date'].max().date()  # Convert to Python's native date

    min_date1 = data['Contract start date'].min().date()  # Convert to Python's native date
    max_date1 = data["Contract start date"].max().date()  # Convert to Python's native date

    min_date2 = data['Contract end date'].min().date()  # Convert to Python's native date
    max_date2 = data['Contract end date'].max().date()  # Convert to Python's native date

    col1, col2, col3 = st.columns(3)



    # Streamlit double-ended slider
    with col1:
        selected_range_creation = st.slider(
            'Contract creation date',
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date)
        )
    with col2:
        selected_range_start= st.slider(
            'Contract start date',
            min_value=min_date1,
            max_value=max_date1,
            value=(min_date1, max_date1)
        )
    with col3:
        selected_range_end = st.slider(
            'Contract end date',
            min_value=min_date2,
            max_value=max_date2,
            value=(min_date2, max_date2)
        )

    filtered_df = data.loc[
        (data['Contract creation date'] >= pd.Timestamp(selected_range_creation[0])) &
        (data['Contract creation date'] <= pd.Timestamp(selected_range_creation[1])) &
        (data['Contract start date'] >= pd.Timestamp(selected_range_start[0])) &
        (data['Contract start date'] <= pd.Timestamp(selected_range_start[1])) &
        (data['Contract end date'] >= pd.Timestamp(selected_range_end[0])) &
        (data['Contract end date'] <= pd.Timestamp(selected_range_end[1]))
        ]

    # Display the filtered data
    st.write(filtered_df.head())

    st.columns(3)
    unique_scores = sorted(filtered_df['scores'].unique())
    with col1:
        selected_score = st.selectbox("Select a score to filter:", unique_scores)


    # slider_range = st.slider("Start Date",  value = [data["Contract creation date"].min(),data["Contract creation date"].max()])
    # st.write("SLider range", slider_range,slider_range[0], slider_range[1] )
    # end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

