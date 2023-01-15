#!/usr/bin/env python
# coding: utf-8

# <img src="mmu.png" style="height: 80px;" align=left> 

# ## Load Libraries

import streamlit as st
import altair as alt
import requests

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from boruta import BorutaPy
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

from apyori import apriori
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

import geopy.distance
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

st.header("Project")

st.subheader("Project Members:")
st.write("(1) 1181102924 Timothy Chang Qing Jun")
st.write("(2) 1181103257 Lye Hong Zheng Marcus")
st.write("(3) 1181103302 Foo Zhi Yao")
st.write("(4) 1181103285 Wong Qian Qing")

st.header("Dataset")
df = pd.read_csv('dataset.csv')
st.write("The original dataset of the project (dataset.csv) is shown below:")
df

df = df.drop(columns=['Basket_colour', 'Shirt_Colour', 'shirt_type', 'Pants_Colour', 'pants_type', 'Attire', 'Spectacles'])


st.header("Additional dataset 1: Weather")

st.write("The weather extra dataset is obtained from the API from the website https://open-meteo.com/en/docs/historical-weather-api. Weather information such as weather code, humidity, temperature and rainfall dated from 01/01/2015 to 31/12/2016 is extracted and downloaded as a csv file.  The chosen location of the weather data is based on the means of the latitudes and longitudes of the dataset, thus obtaining an average of the weather of the area.")

wt = pd.read_csv('weather.csv')

newdate = pd.to_datetime(wt['time']).dt.date
newtime = pd.to_datetime(wt['time']).dt.time
wt.insert(0,'Date', newdate)
wt.insert(1,'Time', newtime)
wt.drop(columns=['time'], inplace=True)


dt_1 = pd.to_datetime(wt['Date'].astype(str) + ' ' + wt['Time'].astype(str))
wt.insert(2, 'DateTime', dt_1)


st.write("The details of the weather code is checked from the WMO code table https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM, it is then reassigned using dictionary. The columns are renamed into more simple namings.")

weather_dict = {0:'Sunny',
                1:'Cloudy',
                2:'Cloudy',
                3:'Cloudy',
                51:'Slight drizzle',
                53:'Moderate drizzle',
                61:'Slight rain',
                63:'Moderate rain',
                65:'Heavy rain'}
wt['weather'] = wt['weathercode (wmo code)'].replace(weather_dict)
wt = wt.rename(columns = {'temperature_2m (°C)':'temperature', 'relativehumidity_2m (%)':'humidity', 'rain (mm)':'rainfall', 'weathercode (wmo code)':'weathercode'})

wt_copy = wt.drop(columns=['Date', 'Time', 'weathercode'])
wt_copy

dt = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
df.insert(2, 'DateTime', dt)
df['DateTime'] = df['DateTime'].dt.round('H')


df = pd.merge(df, wt_copy, how='left', on=['DateTime'])
df.head()


st.header("Additional dataset 2: distance")

st.write("A random laundry service shop is selected as the center point and the distance between each of the customers and the laundry shop is calculated.")

#laundry service location: Express Home Laundry, 24, Jalan Helang 15, Bandar Puchong Jaya, 47100 Puchong, Selangor
laun_coords = (3.060292314138968, 101.62708937958328)

df['distance'] = df.apply(lambda x: geopy.distance.geodesic(laun_coords, (x.latitude, x.longitude)).km, axis=1)
df['distance']

st.write("The original dataset is combined with the extra datasets to form the full dataset, shown below:")
df

st.header("Exploratory Data Analysis (EDA)")

st.header("Check for datatypes and columns:")
st.write("The columns names, as well as their respective data types, are shown below:")
df.dtypes

st.header("Check for null values:")

st.write(df.isna().sum())

st.write("The solution implemented is to fill in missing values with the mean or mode of the column, for numerical and categorical data, respectively.")

timespent_mean = df['TimeSpent_minutes'].mean()
df['TimeSpent_minutes'] = df['TimeSpent_minutes'].fillna(value=timespent_mean)

totalspend_mean = df['TotalSpent_RM'].mean()
df['TotalSpent_RM'] = df['TotalSpent_RM'].fillna(value=totalspend_mean)

df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))
st.write("After filling data:")
df

st.header("Check for imbalanced data")
st.write("After manually checking the datatset, two columns stand out as imbalanced: Kids_Category and Basket_Size.")

fig = plt.figure(figsize=(10, 5))
sns.countplot(x='Kids_Category', data = df)
st.pyplot(fig)

fig = plt.figure(figsize=(10, 5))
sns.countplot(x='Basket_Size', data = df)
st.pyplot(fig)

st.header("Check relationship between variables")

st.subheader("EDA Q1: Will with_kids affects buyDrinks?")

df_kidsDrinks = df[['With_Kids', 'buyDrinks']]

fig = plt.figure(figsize=(10, 5))
sns.boxplot(x='With_Kids', y='buyDrinks', data=df_kidsDrinks).set(title='Box plot of With Kids and Buy Drinks')
st.pyplot(fig)

t_stat, p = ttest_ind(df_kidsDrinks.query('With_Kids=="yes"')['buyDrinks'], df_kidsDrinks.query('With_Kids=="no"')['buyDrinks'])
 
st.write("Using t-test, the final p value is: ", round(p,3))

st.markdown("$H_0$: There is no significant difference between the means of buyDrinks for the respective groups of With_Kids.")
st.markdown("After performing t-test, the p-value obtained is 0.91 which is larger than 0.05. Therefore, we fail to reject $H_0$. There is no significant difference between the means of buyDrinks for the respective groups of With_Kids.")

st.subheader("EDA Q2: Is gender and Body_Size independent?")

df_gender_bodysize = df[['Gender', 'Body_Size']]

crosstab = pd.crosstab(df_gender_bodysize['Gender'], df_gender_bodysize['Body_Size'])

chart1 = alt.Chart(df_gender_bodysize.groupby(["Gender", "Body_Size"]).size().reset_index(name="Count")).mark_bar().encode(
    x='Gender:N',
    y='Count:Q',
	color='Body_Size:N'	
)
chart1

contigency= pd.crosstab(df_gender_bodysize['Gender'], df_gender_bodysize['Body_Size'])
st.write(contigency)

c, p, dof, expected = chi2_contingency(contigency)
st.write("Using chi square test, the final p value is: ",round(p,3))

st.markdown("$H_0$: There is no significant relationship between Gender and Body_Size.")
st.markdown("After performing chi square test the p-value obtained is 0.046 which is smaller than 0.05. Therefore, we reject $H_0$. There is a significant relationship between Gender and Body_Size, and thus the two variables are not independent.")

st.subheader("EDA Q3: Is Race & kids_Category independent?")

df_race_kidscat = df[['Race', 'Kids_Category']]

chart2 = alt.Chart(df_race_kidscat.groupby(['Race', 'Kids_Category']).size().reset_index(name="Count")).mark_bar().encode(
    x='Kids_Category:N',
    y='Count:Q',
	color='Race:N'	
)
chart2

contigency= pd.crosstab(df_race_kidscat['Race'], df_race_kidscat['Kids_Category'])
st.write(contigency)

c, p, dof, expected = chi2_contingency(contigency)
st.write("Using chi square test, the final p value is: ",round(p,3))

st.markdown("$H_0$: There is no significant relationship between Race and Kids_Category.")
st.markdown("After performing chi square test the p-value obtained is 0.001 which is smaller than 0.05. Therefore, we reject $H_0$. There is a significant relationship between Race and Kids_Category, and thus the two variables are not independent.")

st.subheader("EDA Q4: Is there significant difference between Basket_Size and Race?")

df_basket = df[['Basket_Size','Race']]
ct = pd.crosstab(index=df_basket['Race'],columns=df_basket['Basket_Size'])
st.write(ct)

chart3 = alt.Chart(df_basket.groupby(['Basket_Size', 'Race']).size().reset_index(name="Count")).mark_bar().encode(
    x='Basket_Size:N',
    y='Count:Q',
	color='Race:N'	
)
chart3

chi2_result = chi2_contingency(ct)
st.write('Using chi square test, the final p value is: ', chi2_result[1])

st.markdown("$H_0$: There is no significant relationship between Basket_Size and Race.")
st.markdown("After performing t-test the p-value obtained is ~0.01 which is smaller than 0.05. Therefore, we reject $H_0$. There is a significant relationship between Basket_Size and Race, and thus the two variables are not independent.")

st.subheader("EDA Q5: Does with_Kids impact the Basket_Size?")

df_kids = df[['With_Kids', 'Basket_Size']]

chart4 = alt.Chart(df_kids.groupby(['With_Kids', 'Basket_Size']).size().reset_index(name="Count")).mark_bar().encode(
    x='With_Kids:N',
    y='Count:Q',
	color='Basket_Size:N'	
)
chart4

contigency= pd.crosstab(df_kids['With_Kids'], df_kids['Basket_Size'])
st.write(contigency)

c, p, dof, expected = chi2_contingency(contigency)
st.write('Using chi square test, the final p value is: ', p)

st.markdown("$H_0$: There is no significant relationship between With_Kids and Basket_Size.")
st.markdown("After performing t-test the p-value obtained is ~0.000001 which is smaller than 0.05. Therefore, we reject $H_0$. There is a significant relationship between With_Kids and Basket_Size, and thus the two variables are not independent.")

st.header("Questions for Predictive Models")

st.header("Question 1: What factors impact the Basket_Size?")

df3 = df.copy()
df3 = df3.drop(columns=['Date', 'Time', 'DateTime'])

col_list = [col for col in df3.columns.tolist() if df3[col].dtype.name == "object"]
df_le = df3[col_list]
df3 = df3.drop(col_list, 1)
df_le = df_le.apply(LabelEncoder().fit_transform)
df_le = pd.concat([df3, df_le], axis = 1)

st.subheader("Feature Selection 1: Boruta")

y = df_le.Basket_Size
X = df_le.drop("Basket_Size", 1)
colnames = X.columns

st.write("Running model...")
rf = RandomForestClassifier(n_jobs = -1,
                            class_weight = "balanced",
                            max_depth = 5)
feat_selector = BorutaPy(rf, n_estimators="auto", random_state=1)
feat_selector.fit(X.values, y.values.ravel())

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
boruta_score = boruta_score.sort_values("Score", ascending = False)

st.write('---------Top 10----------')
st.write(boruta_score.head(10))

st.write('---------Bottom 10----------')
st.write(boruta_score.tail(10))


st.subheader("Model 1: Logistic Regression")

st.write("Top 5 columns from Boruta were used as X.")

y = df_le.Basket_Size
X = df_le.drop("Basket_Size", 1)

cols = boruta_score.Features[0:5]
X = X[cols].copy()

st.write("Running model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=5)
lr = LogisticRegression(multi_class='multinomial')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

st.write('Model accuracy : ', metrics.accuracy_score(y_test, y_pred)*100,'%') 
st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

prob_LR = lr.predict_proba(X_test)
prob_LR = prob_LR[:, 1]

st.subheader("Logistic Regression with SMOTE")


os = SMOTE()
columns = X_train.columns
os_data_X, os_data_y = os.fit_resample(X_train, y_train)

st.write("Running model...")
lr1 = LogisticRegression()
lr1.fit(os_data_X, os_data_y.ravel())
predictions = lr1.predict(X_test)

st.write('Model accuracy : ', metrics.accuracy_score(y_test, predictions)*100,'%')
st.write(pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose())

prob_LR_SMOTE = lr1.predict_proba(X_test)
prob_LR_SMOTE = prob_LR_SMOTE[:, 1]

st.write("ROC curve is plotted to compare the performance between SMOTE and non-SMOTE dataset on Logistic Regression.")

def get_roc_curve(true_y, y_prob):
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    return fpr, tpr

fpr_LR, tpr_LR = get_roc_curve(y_test, prob_LR)
fpr_LR_SMOTE, tpr_LR_SMOTE = get_roc_curve(y_test, prob_LR_SMOTE)

fig = plt.figure(figsize=(10, 5))
plt.plot(fpr_LR, tpr_LR, color='blue', label='Naive Bayes Classifier')
plt.plot(fpr_LR_SMOTE, tpr_LR_SMOTE, color='red', label='Naive Bayes Classifier with SMOTE') 

plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
st.pyplot(fig)

st.write("No significant difference can be found, so the effect of SMOTE is not very significant for this subset of the dataset, as none of the 5 features used have imbalanced data.")

st.subheader("Model 2: Random Forest")

st.write("Top 5 columns from Boruta were used as X.")

y = df_le.Basket_Size
X = df_le.drop("Basket_Size", 1)

cols = boruta_score.Features[0:5]
X = X[cols].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

st.write("Running model...")
clf = RandomForestClassifier(n_estimators = 100)

clf.fit(X_train, y_train)
  
y_pred = clf.predict(X_test)

st.write("Model accuracy: ", metrics.accuracy_score(y_test, y_pred)*100,'%')

prob_RF = clf.predict_proba(X_test)
prob_RF = prob_RF[:, 1]

auc_RF = roc_auc_score(y_test, prob_RF)

st.subheader("Hyperparameter tuning using GridSearchCV")
st.write("As RandomForest model has various hyperparameters, GridSearchCV was used to find the most optimal set of hyperparameters for the model.")

params_rf = {'n_estimators': [200, 300, 400, 500], 
             'max_depth': [2, 3, 4, 7, 8], 
             'min_samples_leaf': [0.1, 0.2, 0.3],
             'max_features':['log2', 'sqrt'],
             'random_state':[1, 3, 5, 10]}
st.write("Below are the lists of parameters:")
st.write(str(params_rf).replace('\'','\"').replace('], "',']  \n"').replace('{', '').replace('}', '').replace('[', '').replace(']', ''))

st.write("Running model...")
grid_rf = GridSearchCV(estimator=clf,
                      param_grid=params_rf,
                      scoring='neg_mean_squared_error',
                      cv=3,
                      verbose=1,
                      n_jobs=-1)

grid_rf.fit(X_train, y_train)

best_hyperparams = grid_rf.best_params_ 
st.write('Best hyperparameters:')
st.write(str(best_hyperparams).replace('\'','\"').replace(', "','  \n"').replace('{', '').replace('}', ''))

best_model = grid_rf.best_estimator_

y_pred = best_model.predict(X_test)

st.write("Model accuracy (with hyperparameter tuning): ", metrics.accuracy_score(y_test, y_pred)*100,'%')

st.subheader("Model 3: Naive Bayes")
 
st.write("'weather' column is used as X in order to see if weather alone can accurately predict Basket_Size.")

y = df_le.Basket_Size
X = df_le.drop("Basket_Size", 1)
X = X[['weather']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
st.write("Running model...")
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
st.write('Model accuracy: ', nb.score(X_test, y_test)*100,'%')

prob_NB = nb.predict_proba(X_test)
prob_NB = prob_NB[:, 1]

auc_NB = roc_auc_score(y_test, prob_NB)


st.subheader("Hyperparameter Tuning using GridSearchCV")

st.write("As Naive Bayes model has a smoothing hyperparameter, GridSearchCV was used to find the most optimal smooothing hyperparameter value for the model.")
st.markdown("A range of 100 numbers from $10^{-9}$ to to 1 is given to GridSearchCV, spaced evenly on a logarithmic scale.")


param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
}

st.write("Running model...")
nbModel_grid = GridSearchCV(estimator=GaussianNB(), 
                            param_grid=param_grid_nb, 
                            verbose=1, 
                            cv=10, 
                            n_jobs=-1)
nbModel_grid.fit(X_train, y_train)
y_pred = nbModel_grid.predict(X_test)

st.write("Model accuracy (with hyperparameter tuning): ", metrics.accuracy_score(y_test, y_pred)*100,'%')

st.subheader("Model 4: Stacking Model")

y = df_le.Basket_Size
X = df_le.drop("Basket_Size", 1)

def get_stacking():
    level0 = list()
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('bayes', GaussianNB()))
       
    level1 = RandomForestClassifier()  
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model

st.write("Running model...")
ec = get_stacking()
ec.fit(X_train, y_train)
y_pred = ec.predict(X_test)

st.write("Model accuracy (Stacking Model): ", metrics.accuracy_score(y_test, y_pred)*100,'%')

# Calculate AUC
prob_EC = ec.predict_proba(X_test)
prob_EC = prob_EC[:, 1]

auc_EC = roc_auc_score(y_test, prob_EC)

st.subheader("Comparison of models")
st.write("The ROC curve of the above models are compared against each other.")

fpr_NB, tpr_NB = get_roc_curve(y_test, prob_NB)
fpr_RF, tpr_RF = get_roc_curve(y_test, prob_RF)
fpr_EC, tpr_EC = get_roc_curve(y_test, prob_EC)

fig = plt.figure(figsize=(10, 5))

plt.plot(fpr_NB, tpr_NB, color='orange', label='Naive Bayes') 
plt.plot(fpr_RF, tpr_RF, color='blue', label='Random Forest Classifier')
plt.plot(fpr_EC, tpr_EC, color='red', label='Stacking Ensemble Classifer') 

plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()

st.pyplot(fig)
st.write("Comparison of AUC score:")
st.write("AUC of Random Forest model: ", auc_RF*100,'%')
st.write("AUC of Naive Bayes model: ", auc_NB*100,'%')
st.write("AUC of Stacking model: ", auc_EC*100,'%')

st.write("The difference between the final testing accuracy of the models are generally not visibly significant.")
st.write("However, Random Forest model usually has a higher AUC score, suggesting that Random Forest model is geneally more reliable in predictions.")
st.write("Note that actual results shown here may differ due to random states.")

st.header("Question 2: Does weather impact the sales (TotalSpent_RM)?")

st.subheader("Model 1: Linear Regression")

df3 = df.copy()
df3 = df3.drop(columns=['Date', 'Time', 'DateTime'])

col_list = [col for col in df3.columns.tolist() if df3[col].dtype.name == "object"]
df_lr = df3[col_list]
df3 = df3.drop(col_list, 1)
df_lr = df_lr.apply(LabelEncoder().fit_transform)
df_lr = pd.concat([df3, df_lr], axis = 1)

df_lr = df_lr[['temperature_2m (?C)', 'humidity', 'rainfall', 'weather', 'TotalSpent_RM']]
st.write("Below is the dataset used for the model, only taking the weather data with TotalSpent_RM:")
df_lr

y = df_lr.TotalSpent_RM
X = df_lr.drop("TotalSpent_RM", 1)


st.subheader("Feature Selection 2: Correlation Filter Method")
st.write("The columns will be evaluated by correlation matrix. The columns that are highly correlated will be dropped as it is not useful to have multiple columns that are too similar to each other.")

cor_matrix = X.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))

to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.65)]
st.write("The dropped column(s):")
st.write(str(to_drop).replace('[','').replace(']',''))

X = X.drop(columns=to_drop)

st.write("Running model...")
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

st.write(model.summary())

st.header("Association Rule Mining (Shirt color, type, pants color, type and attire)")

st.write("Association Rule Mining is used to find relationships that are common between shirt color, shirt type, pants color, pants type, as well as attire.")

df_asm = pd.read_csv('dataset.csv')
df_asm = df_asm[['Shirt_Colour', 'shirt_type', 'Pants_Colour', 'pants_type', 'Attire']].copy()
df_asm = df_asm.apply(lambda x: x.fillna(x.value_counts().index[0]))
#df_asm.head(10)

df_asm = df_asm.assign(Shirt_Colour = df_asm.Shirt_Colour + ' shirt')
df_asm = df_asm.assign(Pants_Colour = df_asm.Pants_Colour + ' pants')
df_asm = df_asm.assign(pants_type = df_asm.pants_type + ' pants')
st.write("Dataset for association rule mining:")
df_asm

records = []

for i in range(0,3370):
    records.append([str(df_asm.values[i,j]) for j in range(0,5)])
#records[0:3]

association_results = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_results)
#association_results

#print(len(association_results))
#print(association_results[0])


st.write("The results of the assocation rule mining are shown below:")

cnt = 0

for item in association_results:
    cnt += 1
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    st.write("(Rule " + str(cnt) + ") " + items[0] + " -> " + items[1])

    #second index of the inner list
    st.write("Support: " + str(round(item[1],3)))

    #third index of the list located at 0th
    #of the third index of the inner list

    st.write("Confidence: " + str(round(item[2][0][2],4)))
    st.write("Lift: " + str(round(item[2][0][3],4)))
    st.write("=====================================")


st.header("Clustering (lat, lon)")

st.write("The KMeans clustering model will be used to cluster the customer's location based on latitude and longitude, to see how many clusters of customers there are, and where most of the customers are located.")

X = df[['latitude','longitude']]
st.write("Dataset for clustering:")
st.write(X)

st.write("An elbow curve graph is plotted to determine the best k value.")

st.write("Plotting graph...")
distortions = []
for i in range(1, 10):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(X)
    distortions.append(km.inertia_)

# plot
fig = plt.figure(figsize=(10, 5))
plt.plot(range(1, 10), distortions, marker='x')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('Elbow Curve')
plt.show()
st.pyplot(fig)
st.write("From the graph, we can determine that after k=3, the reduction in distortion is less significant. Thus, we choose k=3.")
st.write("Thus, the KMeans algorithm is run with k=3 and then the clusters are assigned.")

st.write("Running model...")
kmeans = KMeans(n_clusters = 3, init ='random', random_state=0)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
X['cluster'] = labels
st.write("Below is the dataset with the cluster labels:")
X

st.write("A scatter plot is plotted to visualise the clusters.")

chart5 = alt.Chart(X).mark_circle(size=60).encode(
	x=alt.X("latitude",scale=alt.Scale(zero=False)),
	y=alt.Y("longitude",scale=alt.Scale(zero=False)),
	color="cluster",
	
)
chart5
