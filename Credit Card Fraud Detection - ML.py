#!/usr/bin/env python
# coding: utf-8

# In[31]:


# Step 1: Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


data = load_and_preview("creditcard.csv")

# Plot settings
sns.set(style="darkgrid", palette="Set2")
plt.rcParams["figure.figsize"] = (12, 6)
warnings.filterwarnings("ignore")



# In[38]:


# Step 2: Load and Preview the Dataset
def load_and_preview(file_path):
    df = pd.read_csv(file_path)
    print(f"✅ Dataset Loaded: {df.shape[0]} records & {df.shape[1]} features\n")
    display(df.head())
    return df


# In[35]:


# Step 3: Descriptive Statistics
def dataset_summary(df):
    print("📊 Descriptive Statistics:")
    display(df.describe().style.background_gradient(cmap="YlGnBu"))

dataset_summary(data)


# In[36]:


# Step 4: Class Distribution Visualization
def plot_class_distribution(df):
    fraud_count = df["Class"].value_counts()
    labels = ["Valid", "Fraud"]
    colors = ["#66bb6a", "#ef5350"]
    
    plt.figure(figsize=(8, 5))
    plt.pie(fraud_count, labels=labels, colors=colors, autopct="%1.4f%%", startangle=140, explode=[0, 0.1], shadow=True)
    plt.title("💳 Distribution of Transaction Classes")
    plt.axis("equal")
    plt.show()

    fraud = df[df["Class"] == 1]
    valid = df[df["Class"] == 0]
    outlier_fraction = len(fraud) / float(len(valid))
    
    print(f"📌 Fraud Ratio: {outlier_fraction:.6f}")
    print(f"🔴 Fraud Cases: {len(fraud)}")
    print(f"🟢 Valid Transactions: {len(valid)}")
    
    return fraud, valid

fraud, valid = plot_class_distribution(data)


# In[37]:


# Step 5: Explore Transaction Amounts
def compare_transaction_amounts(fraud, valid):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(y=fraud["Amount"], ax=axs[0], color="#ef5350")
    axs[0].set_title("Fraudulent Transactions")
    axs[0].set_ylabel("Amount ($)")
    
    sns.boxplot(y=valid["Amount"], ax=axs[1], color="#66bb6a")
    axs[1].set_title("Valid Transactions")
    axs[1].set_ylabel("Amount ($)")
    
    plt.suptitle("💰 Comparison of Transaction Amounts")
    plt.tight_layout()
    plt.show()
    
    print("💡 Summary of Fraud Amounts:\n", fraud["Amount"].describe())



# In[28]:


# Step 6: Correlation Matrix Heatmap
def plot_corr_matrix(df):
    corr = df.corr()
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=False, fmt=".2f", square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("📈 Feature Correlation Matrix (Triangular View)", fontsize=16)
    plt.show()

plot_corr_matrix(data)


# In[29]:





# In[ ]:





# In[ ]:




