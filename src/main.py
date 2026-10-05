import pandas as pd
import re
import matplotlib.pyplot as plt

file_path = "data/raw/jobs.json"
jobs = pd.read_json(file_path)
print(jobs.head())
print(jobs.columns)
print("\nDataset size:")
print(jobs.shape)

print("\nDuplicate rows:")
print(jobs.duplicated().sum())
print("\nWorkplace types:")
print(jobs["workplace_type"].value_counts(dropna=False))

print("\nEmployment types:")
print(jobs["employment_type"].value_counts(dropna=False))

jobs["company"] = jobs["company"].fillna("Unknown")
jobs["salary_range"] = jobs["salary_range"].fillna("Unknown")
jobs["workplace_type"] = jobs["workplace_type"].fillna("Unknown")
jobs["employment_type"] = jobs["employment_type"].fillna("Unknown")

print("\nMissing values:")
print(jobs.isnull().sum())

skills = [
    "Python",
    "SQL",
    "Java",
    "C++",
    "R",
    "Machine Learning",
    "Artificial Intelligence",
    "AI",
    "Generative AI",
    "LLM",
    "PyTorch",
    "TensorFlow",
    "scikit-learn",
    "NLP",
    "Computer Vision",
    "AWS",
    "Azure",
    "GCP",
    "Tableau",
    "Power BI",
    "Excel",
    "Git",
    "Docker"
]

def count_skill(skill):
    if skill == "C++":
        pattern = r"(?<!\w)C\+\+(?!\w)"
    else:
        pattern = r"\b" + re.escape(skill) + r"\b"

    return jobs["description"].str.contains(
        pattern, case=False, na=False, regex=True
    ).sum()

skill_counts = []

for skill in skills:
    count = count_skill(skill)

    skill_counts.append({
        "skill": skill,
        "job_count": count
    })

skills_df = pd.DataFrame(skill_counts)

skills_df = skills_df.sort_values(
    by="job_count",
    ascending=False
)

print("\nMost requested skills:")
print(skills_df)
top_skills = skills_df.head(10)
print("\nTop 10 skills:")
print(top_skills)

plt.barh(top_skills["skill"], top_skills["job_count"])

for index, value in enumerate(top_skills["job_count"]):
    plt.text(value + 5, index, str(value))

plt.xlabel("Number of Job Postings")
plt.ylabel("Skill")
plt.title("Top 10 Most Requested Skills")

plt.gca().invert_yaxis()

plt.savefig(
    "data/visualizations/top_10_skills.png",
    bbox_inches="tight"
)

plt.show()

output_path = "data/processed/jobs_cleaned.csv"

jobs.to_csv(output_path, index=False)

print("\nCleaned dataset saved!")