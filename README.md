# Credit-Card-Fraud-Detection---ML
In this project, our goal is to detect fraudulent credit card transactions using machine learning. We begin by importing a few important Python libraries like NumPy, Pandas, Matplotlib, and Seaborn. These tools help us work with large amounts of data, perform calculations, and create visualizations that make it easier to understand patterns in the data.

Next, we load the dataset, which contains around 284,807 credit card transactions, into a Pandas DataFrame. Each transaction is described by 31 features, including the amount spent, the time of the transaction, and 28 hidden (anonymized) variables labeled V1 through V28 to protect sensitive user information. Most importantly, the dataset includes a ‘Class’ column that tells us whether a transaction is fraudulent (1) or not (0). This column is the target we want our model to predict.

After loading the data, we check how many of the transactions are fraud and how many are normal. We find that fraudulent transactions make up only about 0.17% of the dataset. This is called a class imbalance, and it’s one of the biggest challenges in fraud detection. Because fraud is so rare, many machine learning models might just predict “not fraud” for everything and still seem accurate, even though they're missing the real problem.

To understand more about how fraudulent and valid transactions differ, we explore the amounts involved in each type. We compare the distribution of amounts in both categories, and often we find that fraudulent transactions tend to involve higher or unusual amounts of money. This can be an important clue for our model to learn from.

Finally, we generate a correlation matrix, which shows how all the features in the dataset are related to each other. We use a heatmap to visualize these correlations. This step helps us see if any of the features are strongly linked, which can influence which features we choose to train our model with. Strong correlations might show us patterns, while weak or no correlation could mean a feature is less useful.

Each of these steps—understanding the data, analyzing distributions, and exploring relationships between variables—helps us to recognize fraud.
