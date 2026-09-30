import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Built-in Titanic dataset load kar rahe hain
df = sns.load_dataset("titanic")

# ----------------------------------------------------
# 1. DATA CLEANING
# ----------------------------------------------------

# Age column ke missing values ko Median se fill kar rahe hain
df["age"].fillna(df["age"].median(), inplace=True)

# Embarked column ke missing values ko Mode (most frequent) se fill kar rahe hain
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

# Cabin column mein bohot zyada missing data hota hai, toh usko drop kar dete hain
if "deck" in df.columns:
    df.drop(columns=["deck"], inplace=True)

# ----------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION
# ----------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Survival Rate by Gender
sns.barplot(x="sex", y="survived", data=df, ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("Survival Rate by Gender")
axes[0, 0].set_ylabel("Survival Rate")

# Plot 2: Survival Rate by Passenger Class (Pclass)
sns.barplot(x="pclass", y="survived", data=df, ax=axes[0, 1], palette="Set1")
axes[0, 1].set_title("Survival Rate by Passenger Class")
axes[0, 1].set_ylabel("Survival Rate")

# Plot 3: Age Distribution by Survival Status
sns.histplot(
    data=df,
    x="age",
    hue="survived",
    kde=True,
    bins=25,
    ax=axes[1, 0],
    palette="coolwarm",
)
axes[1, 0].set_title("Age Distribution by Survival Status")

# Plot 4: Correlation Heatmap (Numeric Features)
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(
    numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1, 1]
)
axes[1, 1].set_title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()
