import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def load():
    data = pd.read_csv("WEEK_6/ÖDEVLER/DIABETES/diabetes.csv")
    return data

df = load()


df_=df.copy() #boş değerlere atadığım değerleri kontrol etmek için.

# Step 1: Examine the data.

def check_df(dataframe, head=5):
    print("############### shape #############")
    print(dataframe.shape)
    print("############### types #############")
    print(dataframe.dtypes)
    print("############### head #############")
    print(dataframe.head())
    print("############### tail #############")
    print(dataframe.tail())
    print("############### NA #############")
    print(dataframe.isnull().sum())
    print("############### Quantiles #############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# Step 2: Finding numeric and categorical variables.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
   Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
   Parameters
   ----------
   dataframe: dataframe (type yazıyoruz)
         değişken isimleri alınmak isenen dataframe'dir.
   cat_th:int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
   car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri.
   Returns
   -------
        cat_cols: list
            kategorik değişken listesi
        num_cols: list
            numerik değişken listesi
        cat_but_car:list
            kategorik görünümlü kardinal değişken listesi
   Notes
   ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_car cat_cols'un içerisinde.
    Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car
   """
    cat_cols = [col for col in df.columns if
                str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes in ["int",
                                                                                              "float"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and str(df[col].dtypes) in ["category",
                                                                                                   "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

print(f"cat_cols: {cat_cols}")
print(f"num_cols: {num_cols}")
print(f"cat_but_car: {cat_but_car}")
print(f"num_but_cat: {num_but_cat}")

# Step 3:  Analyzing of the numeric and categorical variables.

def cat_summary(dataframe, col_name, plot=False):  # create plot graph
    if df[col_name].dtypes == "bool":
        df[col_name] = df[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        if plot:  # meaning that plot is true
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:  # meaning that plot is true
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# Step 4: Analyzing target variable. (meaning of the target variable by categorical variables, meaning of numerical variables by target variables)



def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN":dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df,"Outcome",col)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}),end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


# Step 5: Outlier Analyzing

#How to find ?
#1. industry knowledge
#2. standard deviation approach
#3.z-score approach
#4.boxplot(interquantile range -IQR) Method =>
#5. lof yöntemi => multiple variables method

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

#lof
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-') #x gözlemler, y onların outlier scoreları.
plt.show()

th = np.sort(df_scores)[10]

df[df_scores<th].shape

df[df_scores<th]

# Step 6: Missing observation analysis.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

# Steo 7: Correlation Analysis.


corr = df[num_cols].corr()

f, ax = plt.subplots(figsize=[10, 10])
sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# Task 2: Feature Engineering

# Steo 1:  Take necessary actions for missing and outlier values.
# There are no missing observations in the data set, but Glucose, Insulin etc. Observation units containing a value of 0 in the variables may represent the missing value.
# Considering this situation, you can assign the zero values ​​to the relevant values ​​as NaN and then apply the operations to the missing values.

df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]] = df[["Glucose","BloodPressure","SkinThickness", "Insulin","BMI"]].replace(0,np.NaN)
df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_columns)

msno.matrix(df)
plt.show()

msno.heatmap(df) # missing values correlation
plt.show()

# Delete
# value assignment methods(mod, median)
# Predictive methods (ML, statistical methods)


#standardization
scaler = MinMaxScaler()  # Allows converting values ​​from 1 to 0.
df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns) # convert into dataframe
df.head()

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5) # 5 closest neighbors.
df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)
df.head()
df.isnull().sum()
#I'm restoring it to compare. (before standardization)
df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)

df_["Insulin_knn"] = df[["Insulin"]]
df_.loc[df_["Insulin"] == 0, :].head(10)

# Step 2: Creating new variables.

df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "middle_age"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"


df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],labels=["Underweight", "Healthy", "Overweight", "Obese"])


df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"


df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"



def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

from statsmodels.stats.proportion import proportions_ztest
df.groupby("NEW_INSULIN_SCORE").agg({"Outcome": "mean"})


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_INSULIN_SCORE"] == "Normal", "Outcome"].sum(),
                                             df.loc[df["NEW_INSULIN_SCORE"] == "Abnormal", "Outcome"].sum()],

                                      nobs=[df.loc[df["NEW_INSULIN_SCORE"] == "Normal", "Outcome"].shape[0],
                                            df.loc[df["NEW_INSULIN_SCORE"] == "Abnormal", "Outcome"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

df["NEW_GLUCOSE_INSULIN"] = df["Glucose"] * df["Insulin"]


df["NEW_GLUCOSE_PREGNANCIES"] = df["Glucose"] * (1+ df["Pregnancies"])

df.head()
df.shape
# Step 3:  Encoding operations

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

print(f"cat_cols: {cat_cols}")
print(f"num_cols: {num_cols}")
print(f"cat_but_car: {cat_but_car}")

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2] # boş değerleride saydığı için len(unique) almadık.

for col in binary_cols:
    label_encoder(df, col)

df.head()
df["NEW_GLUCOSE"]

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "Outcome", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy() # kopya alınmış.

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)] # kategorik değişken ve oranı 0.01 ise bu değerleri getir.

    for var in rare_columns: #rare column'larda gezilmiş.
        tmp = temp_df[var].value_counts() / len(temp_df)  #rare_column'ın oranı belirlenmiş.
        rare_labels = tmp[tmp < rare_perc].index # index buluyoruz.
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
        # eğer rare_columns'larda gezdiğin değerler rare_labels'da var ise rare yaz, yoksa aynı şekilde bırak.
    return temp_df

df = rare_encoder(df, 0.01)



rare_analyser(df, "Outcome", cat_cols)

def one_hot_encoder(dataframe, categorical_cols, drop_first= True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first = drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique()> 2]

df = one_hot_encoder(df, ohe_cols)
df.head()
df.shape

# Step 4: Standardize for numeric variables.

scaler = MinMaxScaler() # değerleri 1 ile 0 'a dönüştür yapmayı sağlıyor.
df[num_cols]= pd.DataFrame(scaler.fit_transform(df[num_cols]), columns = df[num_cols].columns)
df.head()

# Step 5: Creating a model

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=20)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# How to improve this model?
# changing hyper parameters.
# improving to eliminate missing values methods.
# values which we find in localoutlierfactor are deleted or suppressed the values.
# using different ML methods.
# creating several variables.


# imprtance of variables.

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)
