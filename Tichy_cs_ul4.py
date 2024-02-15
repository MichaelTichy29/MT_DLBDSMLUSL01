# 1. Load the data
# 2. clean and categorize the data
# 3. Use variables of the sets 0 
# 4. Split the data X,Y and Test/Training
# 5. a. analyze the data 
# 5. b. analyze the data, drop mh 
# 5. c. analyze the data drop mh, own treatment_mh


#  feature importance.
import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier




with open('head_of_columns.csv') as namen:
    namelist = csv.reader(namen, delimiter=',') 
    for zeile in namelist:
        namenliste = zeile
df = pd.read_csv('mental_health_c.csv', encoding = 'cp850', names = namenliste)


print("original data set")
print(df.shape)
print("") 


############################
# Cleaning
############################



# Insert "No Answer" to empty cells
for c in df.columns:
    df[c] = df[c].fillna("No Answer")


# cleaning of gender
df.gender = df.gender.replace([' Female', 'Cis-woman', 'Cis female ', 'Cisgender Female', 'Female', 'Female ',
 'Female (props for making this a freeform field, though)', 'Female assigned at birth ', 'Female or Multi-Gender Femme',
 'Genderfluid (born female)', 'I identify as female.', 'f',
 'fem', 'female', 'female ', 'female-bodied; no feelings about gender',
 'female/woman', 'fm', 'woman', 'Woman'],'F')

df.gender = df.gender.replace(['Cis Male', 'Cis male', "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ",
 'MALE', 'Male', 'Male ', 'Male (cis)', 'Male.',
 'Malr', 'Man', 'M|', 'Sex is male', 'cis male', 'cis man', 'm', 'mail', 'male', 'male ',
 'male 9:1 female, roughly', 'man'], 'M')        

df.gender = df.gender.replace(['AFAB', 'Agender', 'Androgynous', 'Bigender', 'Dude', 'Enby', 'Fluid',
 'Genderfluid', 'Genderflux demi-girl', 'Genderqueer', 'Human', 'Male (trans, FtM)', 'Male/genderqueer', 'Nonbinary', 'Other',
 'Other/Transfeminine', 'Queer', 'Transgender woman', 'Transitioned, M2F',
 'Unicorn', 'cisdude', 'genderqueer', 'genderqueer woman', 'human', 'mtf',
 'nb masculine', 'non-binary', 'none of your business', 'No Answer'], 'other')        
        
df.gender = df.gender.fillna("other")

#cleaning of age
df = df.drop(df[df['age']  == 3].index)
df = df.drop(df[df['age']  > 90].index)

#categorisation of ages
df.loc[df.age < 25, 'agec'] = "lower 25"
df.loc[df.age >= 25, 'agec'] = "25 - 29"
df.loc[df.age >= 30, 'agec'] = "30 - 34"
df.loc[df.age >= 35, 'agec'] = "35 - 39"
df.loc[df.age >= 40, 'agec'] = "40 - 49"
df.loc[df.age >= 50, 'agec'] = "upper 50"


# Split the country in us - no us
df['countryc'] = "No US"
df['work_countryc'] = "No US"
df.loc[df.country == "United States of America", 'countryc'] = "US"
df.loc[df.work_country == "United States of America", 'work_countryc'] = "US"


df['mh'] = "No"
df.loc[df.own_mh_in_past == "Yes", 'mh'] = "Yes"
df.loc[df.own_mh_current == "Yes", 'mh'] = "Yes"


######################
# coding of variables
######################
df.ph_issue_in_job_interview = df.ph_issue_in_job_interview.replace(['Yes'], 1)
df.ph_issue_in_job_interview = df.ph_issue_in_job_interview.replace(['No'], 0)
df.ph_issue_in_job_interview = df.ph_issue_in_job_interview.replace(['Maybe'], 0.5)
#
df.mh_issue_in_job_interview = df.mh_issue_in_job_interview.replace(['Yes'], 1)
df.mh_issue_in_job_interview = df.mh_issue_in_job_interview.replace(['No'], 0)
df.mh_issue_in_job_interview = df.mh_issue_in_job_interview.replace(['Maybe'], 0.5)
#
df.mh_issue_neg_career = df.mh_issue_neg_career.replace(['No, it has not'], 0)
df.mh_issue_neg_career = df.mh_issue_neg_career.replace(['No, I don\'t think it would'], 0.25)
df.mh_issue_neg_career = df.mh_issue_neg_career.replace(['Maybe'], 0.5)
df.mh_issue_neg_career = df.mh_issue_neg_career.replace(['Yes, I think it would'], 0.75)
df.mh_issue_neg_career = df.mh_issue_neg_career.replace(['Yes, it has'], 1)
#
df.neg_view_coworker = df.neg_view_coworker.replace(['No, they do not'], 0)
df.neg_view_coworker = df.neg_view_coworker.replace(['No, I don\'t think they would'], 0.25)
df.neg_view_coworker = df.neg_view_coworker.replace(['Maybe'], 0.5)
df.neg_view_coworker = df.neg_view_coworker.replace(['Yes, I think they would'], 0.75)
df.neg_view_coworker = df.neg_view_coworker.replace(['Yes, they do'], 1)
#
df.mh_share_fam = df.mh_share_fam.replace(['Not open at all'], 0)
df.mh_share_fam = df.mh_share_fam.replace(['Somewhat not open'], 0.25)
df.mh_share_fam = df.mh_share_fam.replace(['Somewhat open'], 0.75)
df.mh_share_fam = df.mh_share_fam.replace(['Neutral'], 0.5)
df.mh_share_fam = df.mh_share_fam.replace(['Not applicable to me (I do not have a mental illness)'], 0.5)
df.mh_share_fam = df.mh_share_fam.replace(['Very open'], 1)
#
df.neg_respons_on_mh = df.neg_respons_on_mh.replace(['No'], 0)
df.neg_respons_on_mh = df.neg_respons_on_mh.replace(['Maybe/Not sure'], 0.25)
df.neg_respons_on_mh = df.neg_respons_on_mh.replace(['No Answer'], 0.5)
df.neg_respons_on_mh = df.neg_respons_on_mh.replace(['Yes, I experienced'], 0.75)
df.neg_respons_on_mh = df.neg_respons_on_mh.replace(['Yes, I observed'], 1)
#
df.mh_in_family = df.mh_in_family.replace(['No'], 0)
df.mh_in_family = df.mh_in_family.replace(['I don\'t know'], 0.5)
df.mh_in_family = df.mh_in_family.replace(['Yes'], 1)
#
df.mh = df.mh.replace(['No'], 0)
df.mh = df.mh.replace(['Yes'], 1)
#
df.own_diag_mh = df.own_diag_mh.replace(['No'], 0)
df.own_diag_mh = df.own_diag_mh.replace(['Yes'], 1)
#
df.gender = df.gender.replace(['F'], 0)
df.gender = df.gender.replace(['M'], 1)
df.gender = df.gender.replace(['other'], 0.5)
#
df.countryc = df.countryc.replace(['US'], 0)
df.countryc = df.countryc.replace(['No US'], 1)
#
df.work_countryc = df.work_countryc.replace(['US'], 0)
df.work_countryc = df.work_countryc.replace(['No US'], 1)
#
df.remote = df.remote.replace(['Always'], 1)
df.remote = df.remote.replace(['Never'], 0)
df.remote = df.remote.replace(['Sometimes'], 0.5)




############################
# Selection of the features - set 0
############################


featuretree = ['self_employed','have_prev_emp',
       'ph_issue_in_job_interview',
       'mh_issue_in_job_interview',
        'mh_issue_neg_career', 'neg_view_coworker',
       'mh_share_fam', 'neg_respons_on_mh', 'mh_in_family', 'own_diag_mh', 
       'age', 'gender', 'countryc', 
       'work_countryc', 'remote', 'mh', 'own_treatment_mh']
#  'mh',
# 

aktdf = df
aktfeature = featuretree
raw_aktdf = aktdf[featuretree]
###########################



############################
# Splitting of dataset X,Y and Training, Test
############################


X = raw_aktdf.drop(columns=['own_diag_mh'])
Y = raw_aktdf['own_diag_mh']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.1, random_state = 42)

print('shape of X_train = ', X_train.shape)
print('shape of X_test = ', X_test.shape)
print('shape of Y_train = ', Y_train.shape)
print('shape of Y_test = ', Y_test.shape)

dt= DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dt.fit(X_train, Y_train)
dt_pred_train = dt.predict(X_train)

#evaluate on train
dt_pred_test = dt.predict(X_train)
print('Training set f1 (all variables of set 0) = ', f1_score(Y_train, dt_pred_train))


#evaluate on test
dt_pred_test = dt.predict(X_test)
print('Testingset f1 (all variables of set 0) = ', f1_score(Y_test, dt_pred_test))


feature_importance=pd.DataFrame({
    #'rfc':rfc.feature_importances_,
    'dt':dt.feature_importances_
},index=raw_aktdf.drop(columns=['own_diag_mh']).columns)
#feature_importance.sort_values(by='rfc',ascending=True,inplace=True)

index = np.arange(len(feature_importance))
fig, ax = plt.subplots(figsize=(18,8))
#rfc_feature=ax.barh(index,feature_importance['rfc'],0.4,color='purple',label='Random Forest')
dt_feature=ax.barh(index+0.4,feature_importance['dt'],0.4,color='lightgreen',label='Decision Tree Set 0')
ax.set(yticks=index+0.4,yticklabels=feature_importance.index)

ax.legend()
plt.show()

print('#######################################')


################ drop 'mh' #########################


featuretree = ['self_employed','have_prev_emp',
       'ph_issue_in_job_interview',
       'mh_issue_in_job_interview',
        'mh_issue_neg_career', 'neg_view_coworker',
       'mh_share_fam', 'neg_respons_on_mh', 'mh_in_family', 'own_diag_mh', 
       'age', 'gender', 'countryc', 
       'work_countryc', 'remote', 'own_treatment_mh']
#  'mh',
# 

aktdf = df
aktfeature = featuretree
raw_aktdf = aktdf[featuretree]
###########################


X = raw_aktdf.drop(columns=['own_diag_mh'])
Y = raw_aktdf['own_diag_mh']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.1, random_state = 42)


dt= DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dt.fit(X_train, Y_train)
dt_pred_train = dt.predict(X_train)

#evaluate on train
dt_pred_test = dt.predict(X_train)
print('Training set f1 (drop mh) = ', f1_score(Y_train, dt_pred_train))


#evaluate on test
dt_pred_test = dt.predict(X_test)
print('Testingset f1 (drop mh) = ', f1_score(Y_test, dt_pred_test))


feature_importance=pd.DataFrame({
    #'rfc':rfc.feature_importances_,
    'dt':dt.feature_importances_
},index=raw_aktdf.drop(columns=['own_diag_mh']).columns)
#feature_importance.sort_values(by='rfc',ascending=True,inplace=True)

index = np.arange(len(feature_importance))
fig, ax = plt.subplots(figsize=(18,8))
#rfc_feature=ax.barh(index,feature_importance['rfc'],0.4,color='purple',label='Random Forest')
dt_feature=ax.barh(index+0.4,feature_importance['dt'],0.4,color='lightgreen',label='Decision Tree drop mh')
ax.set(yticks=index+0.4,yticklabels=feature_importance.index)

ax.legend()
plt.show()


print('#######################################')

################ drop 'mh', own_treatment_mh #########################


featuretree = ['self_employed','have_prev_emp',
       'ph_issue_in_job_interview',
       'mh_issue_in_job_interview',
        'mh_issue_neg_career', 'neg_view_coworker',
       'mh_share_fam', 'neg_respons_on_mh', 'mh_in_family', 'own_diag_mh', 
       'age', 'gender', 'countryc', 
       'work_countryc', 'remote']
#  'mh', , 'own_treatment_mh'
# 

aktdf = df
aktfeature = featuretree
raw_aktdf = aktdf[featuretree]
###########################


X = raw_aktdf.drop(columns=['own_diag_mh'])
Y = raw_aktdf['own_diag_mh']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.1, random_state = 42)


dt= DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dt.fit(X_train, Y_train)
dt_pred_train = dt.predict(X_train)

#evaluate on train
dt_pred_test = dt.predict(X_train)
print('Training set f1 (drop mh, own_treatment_mh) = ', f1_score(Y_train, dt_pred_train))


#evaluate on test
dt_pred_test = dt.predict(X_test)
print('Testingset f1 (drop mh, own_treatment_mh) = ', f1_score(Y_test, dt_pred_test))


feature_importance=pd.DataFrame({
    #'rfc':rfc.feature_importances_,
    'dt':dt.feature_importances_
},index=raw_aktdf.drop(columns=['own_diag_mh']).columns)
#feature_importance.sort_values(by='rfc',ascending=True,inplace=True)

index = np.arange(len(feature_importance))
fig, ax = plt.subplots(figsize=(18,8))
#rfc_feature=ax.barh(index,feature_importance['rfc'],0.4,color='purple',label='Random Forest')
dt_feature=ax.barh(index+0.4,feature_importance['dt'],0.4,color='lightgreen',label='Decision Tree drop mh, own_treatment_mh')
ax.set(yticks=index+0.4,yticklabels=feature_importance.index)

ax.legend()
plt.show()
