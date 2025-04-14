# JobRecommendationSystem

A smart and intuitive job recommendation system built with **Streamlit**, designed to help users discover their ideal job opportunities based on their **skills**, **experience**, **location**, and **salary expectations**.

---

## 🚀 Features

- 🔍 Search by Skills, Title, or Company
- 🎯 AI-Powered Recommendations using TF-IDF and Cosine Similarity
- 📊 Real-time Salary Filtering
- 🌍 Filter by Location and Experience Level
- 🧠 Match Score to prioritize best-fit jobs
- 💡 Career Tips and Insights
- 🌙 Clean, responsive dark-themed UI

---

## 📂 Dataset

The app uses a CSV file named `indian_jobs.csv` with the following columns:

- `JobID`
- `Title`
- `Description`
- `RequiredSkills`
- `ExperienceLevel` (Entry, Mid, Senior)
- `Location`
- `Salary` (Annual in INR)

> ✅ Place the CSV in the same folder as `app.py`

---

## 🛠️ Tech Stack

- Python 3.x
- Streamlit
- Pandas
- scikit-learn
- TF-IDF + Cosine Similarity

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/job-recommender.git
cd job-recommender

