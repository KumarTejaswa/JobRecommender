import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set page configuration
st.set_page_config(
    page_title="Job Recommender",
    page_icon="💼",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
    <style>
    body, .main {
        background-color: #1a1a1a;
        font-family: 'Segoe UI', sans-serif;
        color: #ffffff;
    }

    .stMultiSelect > div:first-child {
        display: flex;
    }

    .stSelectbox>div>div {
        flex: 1;
        margin-right: -1px !important;
    }

    .stSelectbox>div>div>div {
        background-color: #ffffff !important;
        color: #212529 !important;
        border: 1px solid #ced4da !important;
        border-right: none !important;
        border-radius: 0px !important;
        position: relative;
    }

    .stSelectbox>div>div:first-child>div {
        border-radius: 5px 0 0 5px !important;
    }

    .stSelectbox>div>div:last-child>div {
        border-radius: 0 5px 5px 0 !important;
        border-right: 1px solid #ced4da !important;
    }

    .stSelectbox>div>div>div::after {
        content: "";
        width: 0;
        height: 0;
        border-left: 6px solid transparent;
        border-right: 6px solid transparent;
        border-top: 6px solid #000000;
        position: absolute;
        right: 10px;
        top: 50%;
        transform: translateY(-50%);
    }

    label[for="experience_level"],
    label[for="location"] {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load CSV
@st.cache_data
def load_data():
    csv_path = "indian_jobs.csv"
    if not os.path.exists(csv_path):
        st.error("CSV file not found! Please make sure 'indian_jobs.csv' is in the same folder as this script.")
        st.stop()

    df = pd.read_csv(csv_path)
    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')

    required_cols = ['JobID', 'Title', 'Description', 'RequiredSkills', 'ExperienceLevel', 'Location', 'Salary']
    if not all(col in df.columns for col in required_cols):
        st.error("CSV missing required columns!")
        st.stop()

    return df

df = load_data()

# TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['RequiredSkills'] + ' ' + df['Description'])

# Recommendation function
def get_recommendations(user_skills, exp_level=None, location=None, min_salary=0, top_n=10):
    user_profile = tfidf.transform([user_skills])
    cos_sim = cosine_similarity(user_profile, tfidf_matrix).flatten()

    df['MatchScore'] = cos_sim
    filtered = df[df['Salary'] >= min_salary].copy()

    if exp_level and exp_level != 'Any':
        filtered = filtered[filtered['ExperienceLevel'] == exp_level]
    if location and location != 'Any':
        filtered = filtered[filtered['Location'] == location]

    return filtered.sort_values('MatchScore', ascending=False).head(top_n)

# UI: Title
st.title("💼 Job Recommender")
st.markdown("### Discover Your Next Career Move with Intelligent Matching")

# User Inputs
with st.container():
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        user_skills = st.text_input("🔧 Enter Your Skills (comma separated):", 
                                    "Python, SQL, Data Analysis",
                                    help="Example: Java, React, Machine Learning")
    with col2:
        exp_level = st.selectbox("📈 Experience Level:", ['Any', 'Entry', 'Mid', 'Senior'])
    with col3:
        location = st.selectbox("📍 Preferred Location:", ['Any'] + sorted(df['Location'].unique()))

# Sidebar Filters
with st.sidebar:
    st.header("⚙️ Filters")
    min_salary = st.slider("💸 Minimum Annual Salary (LPA):", 3, 40, 6) * 100000
    remote_only = st.checkbox("🏠 Show Remote Jobs Only (coming soon)")

    st.markdown("---")
    st.header("💡 Career Tips")
    st.markdown("""
    - 📜 Get certifications (AWS, Scrum, PMP)
    - 💬 Network on LinkedIn
    - 🤖 Contribute to open-source
    - 📚 Practice DSA and system design
    """)

    st.markdown("---")
    st.header("📊 Insights")
    st.markdown(f"""
    - Total Jobs: {len(df):,}
    - Avg Salary: ₹{df['Salary'].mean()/100000:.1f} LPA
    - Top Cities: {', '.join(df['Location'].value_counts().head(2).index)}
    """)

# Recommendation Button
if st.button("🚀 Find Best Matches"):
    with st.spinner("Finding jobs just for you..."):
        results = get_recommendations(
            user_skills=user_skills,
            exp_level=exp_level if exp_level != 'Any' else None,
            location=location if location != 'Any' else None,
            min_salary=min_salary
        )

        if not results.empty:
            st.success(f"✨ Found {len(results)} Matching Opportunities")
            for _, row in results.iterrows():
                with st.expander(f"{row['Title']} | {row['Location']} | ₹{row['Salary']/100000:.1f} LPA", expanded=False):
                    cols = st.columns([2, 1])
                    cols[0].markdown(f"""
                    **Description:**  
                    {row['Description']}

                    **Required Skills:**  
                    {row['RequiredSkills']}
                    """)
                    cols[1].markdown(f"""
                    **Experience:** {row['ExperienceLevel']}  
                    **Match Score:** {row['MatchScore']*100:.1f}%  
                    **Job ID:** {row['JobID']}
                    """)
        else:
            st.warning("No jobs matched. Try adjusting filters.")

# Search all jobs
st.markdown("---")
st.subheader("🔍 Explore All Job Opportunities")
search_term = st.text_input("Search by Job Title, Skills, or Company:")

if search_term:
    matches = df[
        df['Title'].str.contains(search_term, case=False) |
        df['RequiredSkills'].str.contains(search_term, case=False) |
        df['Description'].str.contains(search_term, case=False)
    ]

    if not matches.empty:
        st.write(f"Showing {len(matches)} results for '{search_term}'")
        for _, row in matches.iterrows():
            st.markdown(f"""
            <div class="job-card">
                <h4>{row['Title']}</h4>
                <p><strong>Location:</strong> {row['Location']} | 
                <strong>Experience:</strong> {row['ExperienceLevel']} | 
                <strong>Salary:</strong> ₹{row['Salary']/100000:.1f} LPA</p>
                <p><strong>Skills:</strong> {row['RequiredSkills']}</p>
                <p>{row['Description']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No matching jobs found.")
