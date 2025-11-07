# Demographic Data Analyzer
# Dataset Source: https://archive.ics.uci.edu/ml/datasets/adult
# This dataset contains demographic information such as age, education, occupation, and income.

import pandas as pd

# Load the dataset
df = pd.read_csv("adult.data.csv", header=None, names=[
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"
])

# Clean up whitespace in text columns
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Display dataset info
print("First 5 rows of dataset:")
print(df.head(), "\n")

print("Dataset Info:")
print(df.info(), "\n")

print("Missing Values per Column:")
print(df.isnull().sum(), "\n")

print("Basic Statistics:")
print(df.describe(), "\n")

# Number of each race
race_count = df['race'].value_counts()
print("Number of each race:\n", race_count, "\n")

# Average age of men
average_age_men = round(df[df['sex'] == 'Male']['age'].mean(), 1)
print("Average age of men:", average_age_men, "\n")

# Percentage with Bachelor's degree
percentage_bachelors = round(
    (df['education'].value_counts(normalize=True)['Bachelors'] * 100), 1
)
print(f"Percentage with Bachelors degrees: {percentage_bachelors}%\n")

# Percentage of people with higher education earning >50K
advanced = df['education'].isin(['Bachelors', 'Masters', 'Doctorate'])
higher_education = df[advanced]
lower_education = df[~advanced]

higher_education_rich = round(
    (higher_education[higher_education['salary'] == '>50K'].shape[0] /
     higher_education.shape[0]) * 100, 1
)
lower_education_rich = round(
    (lower_education[lower_education['salary'] == '>50K'].shape[0] /
     lower_education.shape[0]) * 100, 1
)

print(f"Percentage with higher education earning >50K: {higher_education_rich}%")
print(f"Percentage without higher education earning >50K: {lower_education_rich}%\n")

# Minimum work hours per week
min_work_hours = df['hours-per-week'].min()

#  Percentage of rich people who work minimum hours
num_min_workers = df[df['hours-per-week'] == min_work_hours]
rich_percentage = round(
    (num_min_workers[num_min_workers['salary'] == '>50K'].shape[0] /
     num_min_workers.shape[0]) * 100, 1
)

print(f"Minimum work hours per week: {min_work_hours}")
print(f"Percentage of rich among those who work minimum hours: {rich_percentage}%\n")

#  Country with highest percentage of rich people
country_earning = df.groupby('native-country')['salary'].value_counts(normalize=True).unstack()['>50K'] * 100
country_earning = country_earning.dropna()

highest_country = country_earning.idxmax()
highest_percentage = round(country_earning.max(), 1)

print(f"Country with highest percentage of rich: {highest_country}")
print(f"Highest percentage of rich people in that country: {highest_percentage}%\n")

# Most popular occupation for those who earn >50K in India
top_IN_occupation = df[
    (df['native-country'] == 'India') & (df['salary'] == '>50K')
]['occupation'].mode()[0]

print("Most popular occupation for those who earn >50K in India:", top_IN_occupation)
