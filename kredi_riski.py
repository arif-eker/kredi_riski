# Gerekli kütüphaneler ekleniyor.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

import scripts.helper_functions as hlp
import scripts.models as models

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    classification_report

pd.pandas.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: '%.2f' % x)

# Veri okunuyor.

df = pd.read_csv("data/credit_risk.csv")

# Verinin baştan ve sondan ilk 5 gözlemine bakıyoruz. Böylelikle doğru okunmuş mu fikrimiz oluyor.
df.head()
df.tail()

# Verimizi biraz daha yakından tanımamızı yarayan fonksiyondur.
hlp.check_dataframe(df)
df.info()

# Verimizdeki değişkenlerin kaçar adet sınıfı var bakıyoruz.
for col in df.columns:
    print(col, " : ", df[col].nunique(), " eşsiz sınıfa sahip.")

# Anlamsız değişken siliniyor.
df.drop("Unnamed: 0", axis=1, inplace=True)

# 2 sınıfa sahip değişkenlerimizi label encode ile encode edelim.
label_columns = ["Sex", "Risk"]
hlp.label_encoder(df, label_columns)

# Kategorik ve sayısal değişkenlerimizi belirliyoruz.
categorical_columns = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]
numeric_columns = ["Age", "Credit amount", "Duration"]

# Kategorik değişken kırılımında hedef değişkenimizi inceliyoruz.
hlp.cat_summary(df, categorical_columns, "Risk", True)

# Nadir sınıf analizi yapıyoruz.
hlp.rare_analyser(df, categorical_columns, "Risk", 0.5)

# Purpose değişkeninde 1 adet nadir sınıf var.
df.loc[df["Purpose"] == "domestic appliances", ["Purpose"]] = "furniture/equipment"

# Kategorik değişkenlerle işimiz bitti. Sayısal değişkenlerimize bakalım.
hlp.has_outliers(df, numeric_columns)

# Age değişkeninden, Age_Range kategorik değişkeni türetiliyor.
bins = [18, 25, 40, 55, 100]
names = ['Young', 'Adult', 'Mature', 'Old']
df["Age_Range"] = pd.cut(df['Age'], bins, labels=names)

# Duration değişkeninden Year değişkeni türetiliyor.
# Böylece kaç yıllık müşteri olduğu anlaşılacak.
df["Year"] = str(df["Duration"])
df.loc[df["Duration"] <= 12, "Year"] = "0-1 year"
df.loc[(df["Duration"] > 12) & (df["Duration"] <= 24), "Year"] = "1-2 year"
df.loc[(df["Duration"] > 24) & (df["Duration"] <= 36), "Year"] = "2-3 year"
df.loc[(df["Duration"] > 36) & (df["Duration"] <= 48), "Year"] = "3-4 year"
df.loc[(df["Duration"] > 48) & (df["Duration"] <= 60), "Year"] = "4-5 year"
df.loc[(df["Duration"] > 60) & (df["Duration"] <= 72), "Year"] = "5-6 year"
df.loc[(df["Duration"] > 72) & (df["Duration"] <= 84), "Year"] = "6-7 year"

# Status değişkeni Credit amount değişkeninden türeyen, ekonomik sınıfı simgeleyen değişken.
df["Status"] = pd.qcut(df["Credit amount"], 4, labels=["poor", "mid", "upper", "rich"])

# Eksik değerler dolduruluyor.
df["Saving accounts"] = df.groupby(["Sex", "Risk", "Age_Range"])["Saving accounts"].transform(
    lambda x: x.fillna(x.mode()[0]))

df["Checking account"] = df.groupby(["Sex", "Risk", "Age_Range"])["Checking account"].transform(
    lambda x: x.fillna(x.mode()[0]))

# Kategorik değişkenlerimizi modele sokmak için one-hot yapmalıyız.
one_hot_columns = ["Job", "Housing", "Saving accounts", "Checking account", "Purpose", "Age_Range", "Year", "Status"]

df, one_hot_encodeds = hlp.one_hot_encoder(df, one_hot_columns)

# Artık model kurma işlemine geçebiliriz.

X = df.drop("Risk", axis=1)
y = np.ravel(df[["Risk"]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=357)

rf_tuned, lgbm_tuned, xgb_tuned = models.get_tuned_models(X_train, y_train, 357)

# Modellerle tahmin yapıyoruz ve onuçları ekrana veriyoruz.
models = [("RF", rf_tuned),
          ("LGBM", lgbm_tuned),
          ("XGB", xgb_tuned)]

for name, model in models:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    msg = "%s: (%f)" % (name, acc)
    print(msg)
