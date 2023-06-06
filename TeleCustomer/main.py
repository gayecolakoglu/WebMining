#----------------------------
# Import necessary libraries
#----------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier

#------------------------
# ## EDA: General Idea About Data
#------------------------
df = pd.read_csv("Telco-Customer-Churn.csv")

def general_idea(df):
    print("----HEAD:")
    print(df.head())
    print("\n")
    print("----SHAPE:")
    print(df.shape)
    print("\n")
    print("----TYPES:")
    print(df.dtypes)
    print("\n")
    print("----NULL:")
    print(df.isnull().sum())
    print("\n")
    print("----IQR:")
    print(df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
general_idea(df)

# TotalCharges needs to be numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.head()

#Obejct to numeric
df['Churn'] = df['Churn'].apply(lambda x : 1 if x == "Yes" else 0)
df.head()

#-----------------------------------------
# Catch numeric and categorical variables
#-----------------------------------------
def grab_col_names(df, cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if
                   df[col].nunique() < cat_th and df[col].dtypes != "O"]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > car_th and df[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols
num_cols
cat_but_car


# Analyze Categorical Variables
def cat_summary(df, col_name, plot=False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100*df[col_name].value_counts()/len(df)}))

    print("*******************************")
    if plot:
        sns.countplot(x=df[col_name], data=df)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

cat_cols


# Analyze Numeric Variables
def num_summary(df, num_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(df[num_cols].describe(quantiles).T)

    if plot:
        df[num_cols].hist(bins=20)
        plt.xlabel(num_cols)
        plt.title(num_cols)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

num_cols


# Analyze Numeric Variables via Target
def target_summary_with_num(df, target, num_col):
    print(df.groupby(target).agg({num_col:"mean"}),end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)


# Analyze Categorical Variables via Target
def target_summary_with_cat(df, target, cat_col):
    print(cat_col)
    print(pd.DataFrame({"TARGET_MEAN": df.groupby(cat_col)[target].mean(),
                        "Count": df[cat_col].value_counts(),
                        "Ratio": 100*df[cat_col].value_counts() / len(df)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

#--------------
# Correlation
#--------------
df[num_cols].corr()

# Correlation Matrix
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

#We can see that TotalChargers is highly correlated with monthly payment and tenure.
df.corrwith(df["Churn"]).sort_values(ascending=False)

#------------------------
# ## Feature Engineering: Analyze of NaN
#------------------------
df.isnull().sum()

def missing_values_table(df, na_name=False):
    na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (df[na_cols].isnull().sum() / df.shape[0]*100).sort_values(ascending=False)
    df_missing = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(df_missing, end="\n")
    if na_name:
        return na_cols

na_cols = missing_values_table(df, na_name=True)


# Filling Na Values
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df.isnull().sum()

#------------
# BASE MODEL
#------------
dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]
cat_cols
df[cat_cols].head()

def one_hot_encoder(df, cat_cols, drop_first=False):
    df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
    return df

dff = one_hot_encoder(dff, cat_cols, drop_first=True)
dff.head()

y = dff["Churn"]
X = dff.drop(["Churn", "customerID"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"AUC: {round(roc_auc_score(y_pred, y_test), 2)}")

#------------
# Analyze Outliers
#------------
def outlier_th(df, col_name, q1=0.05, q3=0.95):
    quart1 = df[col_name].quantile(q1)
    quart3 = df[col_name].quantile(q3)
    iqr = quart3 - quart1
    up_limit = quart3 + 1.5*iqr
    low_limit = quart1 - 1.5*iqr
    return low_limit, up_limit

def check_outliers(df, col_name):
    low_limit, up_limit = outlier_th(df, col_name)
    if df[(df[col_name] > up_limit) | (df[col_name] > low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_th(df, var, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_th(df, var, q1=0.05, q3=0.95)
    df.loc[(df[var] < low_limit), var] = low_limit
    df.loc[(df[var] > up_limit), var] = up_limit

for col in num_cols:
    print(col, check_outliers(df, col))
    if check_outliers(df, col):
        replace_with_th(df, col)

#------------
# Feature Extraction
#------------

# Create tenure_year categorical variable from 'tenure'
df.loc[(df["tenure"]>=8) & (df["tenure"]<=12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>=12) & (df["tenure"]<=24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>=24) & (df["tenure"]<=36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>=36) & (df["tenure"]<=48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>=48) & (df["tenure"]<=38), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>=68) & (df["tenure"]<=72), "NEW_TENURE_YEAR"] = "5-6 Year"

# Specify customers as engaged who has 1-2 years contract
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

# People who do not receive any support, backup or protection
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Young people with a monthly contract
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

# The total number of services received by the person
df["NEW_TotalServices"] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
'StreamingMovies']] == "Yes").sum(axis=1)

# People who receive any streaming
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Does the person make automatic payments?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)

# Average monthly payment
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Increase in current price relative to average price
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Fee per service
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df["NEW_TotalServices"] + 1)

df.head()
df.shape


#------------
# ENCODING
#------------
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#--LabelEncoding
def label_encoder(df, binary_cols):
    labelencoder = LabelEncoder()
    df[binary_cols] = labelencoder.fit_transform(df[binary_cols])
    return df

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

#--OneHotEncoding
# Update cat_cols list
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
cat_cols

def one_hot_encoder(df, cat_cols, drop_first=False):
    df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
    return df

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.shape
df.head()

#-----------
# MODELING
#-----------
y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 2)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"AUC: {round(roc_auc_score(y_pred, y_test), 2)}")


#--------------------
# FEATURE IMPORTANCE
#--------------------

def plot_feature_importance(importance, names, model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order desc feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(15, 10))
    # Plot Seaborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FATURE NAMES')
    plt.show()

plot_feature_importance(catboost_model.get_feature_importance(), X.columns, 'CATBOOST')
