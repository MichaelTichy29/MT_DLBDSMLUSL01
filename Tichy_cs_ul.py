# 1. Load the data
# 2. Overview to the data
# 3. clean and categorize the data
# 4. analyze the data with the maximum feature set (without text features.)

import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import mixture
from sklearn.metrics import silhouette_score
import seaborn as sns

# load the csv as a pandas dataframe and set new (shorter) names for the variables

with open('head_of_columns.csv') as namen:
    namelist = csv.reader(namen, delimiter=',') 
    for zeile in namelist:
        namenliste = zeile
df = pd.read_csv('mental_health_c.csv', encoding = 'cp850', names = namenliste)


#shape of the original dataset
print("original data set")
print(df.shape)
print("") 

# Test for empty cells
y = df.isnull().sum()
z = df.shape
print(df.shape)
for k in range (0,z[1]): 
    print(y[k])
    print(df.columns[k])
    print("")



# Seperation for self employd - non self employed and first contract.
print("Number of self employed", sum((df.self_employed == 1)))

print("Number of people with an previous job", sum((df.have_prev_emp == 1)))

print("Number of self employed and prev job", sum((df.self_employed == 1) & (df.have_prev_emp == 1)))

print("Number of non self employed and prev job", sum((df.self_employed == 0) & (df.have_prev_emp == 1)))

print("Number of self employed and no prev job", sum((df.self_employed == 1) & (df.have_prev_emp == 0)))

print("Number of non self employed and no prev job", sum((df.self_employed == 0) & (df.have_prev_emp == 0)))

#overview to the number of different answers for each variable
for column in df:
    unique_vals = np.unique(df[column].apply(str))
    nr_vals = len(unique_vals)
    if nr_vals < 10:
        print ("number of values for the feature {}: {} -- {}".format(column, nr_vals, unique_vals))
    else:
        print ("number of values for the feature {}: {}".format(column, nr_vals))


#closer look on the ages, to drop some rows with unreasonable values
unique_vals = np.unique(df.age.apply(str))
nr_vals = len(unique_vals)
print ("number of values for the feature age : {} -- {}".format(nr_vals, unique_vals))

# closer look on the gender to replace the answers
unique_vals = np.unique(df.gender.apply(str))
nr_vals = len(unique_vals)
print ("number of values for the feature gender : {} -- {}".format(nr_vals, unique_vals))


# Insert "No Answer" to empty cells
for c in df.columns:
    df[c] = df[c].fillna("No Answer")



############################
# Cleaning
############################

# cleaning of gender, replacing how the answer is understood
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

#cleaning of age - three unreasonalbe values
df = df.drop(df[df['age']  == 3].index)
df = df.drop(df[df['age']  > 90].index)

#categorisation of ages
df.loc[df.age < 25, 'agec'] = "lower 25"
df.loc[df.age >= 25, 'agec'] = "25 - 29"
df.loc[df.age >= 30, 'agec'] = "30 - 34"
df.loc[df.age >= 35, 'agec'] = "35 - 39"
df.loc[df.age >= 40, 'agec'] = "40 - 49"
df.loc[df.age >= 50, 'agec'] = "upper 50"


# Split the country in the categories  us - no us
df['countryc'] = "No US"
df['work_countryc'] = "No US"
df.loc[df.country == "United States of America", 'countryc'] = "US"
df.loc[df.work_country == "United States of America", 'work_countryc'] = "US"


# tech_role', 'state','work_state', 'position', 'why_ph_issue_in_job_int', 'why_mh_issue_in_job_int',
# 'which_own_diag_mh', 'sure_kind_own_mh', 'maybe_kind_own_mh',
allfeatures = ['self_employed', 'number_employees', 'tech_organisation',
       'employer_provide_mh', 'you_know_options_mh', 'employer_discussed_mh',
       'employer_offer_resources_mh', 'anonymity_protected_mh',
       'able_leave_for_mh', 'discussing_is_negativ_mh',
       'discussing_is_negative_ph', 'like_to_discuss_coworker_mh',
       'like_to_discuss_supervisor_mh', 'employer_ph_to_mh',
       'heard_of_neg_cons_mh', 'med_cov_with_mh', 'know_res_seek_mh',
       'diag_or_treat_mh', 'bus_cont_mh_is_neg_input',
       'reveal_own_mh_to_coworker', 'have_rev_mh_to_coworker_believe_neg',
       'prod_eff_by_mh', 'perc_work_time_eff_by_mh', 'have_prev_emp',
       'prev_emp_prov_mh', 'prev_you_know_options_mh',
       'prev_employer_discussed_mh', 'prev_employer_offer_resources_mh',
       'prev_anonymity_protected_mh', 'prev_discussing_is_negativ_mh',
       'prev_discussing_is_negative_ph', 'prev_like_to_discuss_coworker_mh',
       'prev_like_to_discuss_supervisor_mh', 'prev_employer_ph_to_mh',
       'prev_heard_of_neg_cons_mh', 'ph_issue_in_job_interview',
       'mh_issue_in_job_interview',
        'mh_issue_neg_career', 'neg_view_coworker',
       'mh_share_fam', 'neg_respons_on_mh', 'impact_disuss_mh_other_on_you',
       'mh_in_family', 'own_mh_in_past', 'own_mh_current', 'own_diag_mh', 
       'own_treatment_mh', 'mh_interfer_work_if_treated',
       'mh_interfer_work_if_not_treated', 'agec', 'gender', 'countryc', 
       'work_countryc', 'remote']




"""
features0 = ['self_employed',  'have_prev_emp',
       'ph_issue_in_job_interview',
       'mh_issue_in_job_interview',
       'mh_issue_neg_career', 'neg_view_coworker',
       'mh_share_fam', 'neg_respons_on_mh', 
       'mh_in_family', 'own_mh_in_past', 'own_mh_current','own_diag_mh', 
       'own_treatment_mh', 'agec', 'gender', 'countryc', 
       'work_countryc', 'remote']
"""


    
# second overview to the different answer. Only for the feature set 0. 
for f in allfeatures:
    unique_vals = np.unique(df[f].apply(str))
    nr_vals = len(unique_vals)
    if nr_vals < 10:
        print ("number of values for the feature {}: {} -- {}".format(f, nr_vals, unique_vals))
    else:
        print ("number of values for the feature {}: {}".format(f, nr_vals))


#plot of the counts in the categories
for f in allfeatures:
    sns.countplot(x = f, data = df, palette = 'Set3')
    plt.xticks(rotation=45)
    plt.show()
    


############################
# Coding of the features
############################

raw_aktdf = df[allfeatures]
new_raw_aktdf = pd.get_dummies(data = raw_aktdf, columns = allfeatures)
print("shape of dummy data is")
print(new_raw_aktdf.shape)

x_train = new_raw_aktdf.values




############################
# pca to get 95 % of the variance
############################

#run the pca
n_components = x_train.shape[1]
pca = PCA(n_components=n_components, random_state = 453)
#pca = PCA(n_components=n_components)
x_r = pca.fit(x_train).transform(x_train)



#### 
# 95 % variance
####
tot_var = sum(pca.explained_variance_)
print("total variance is = ", tot_var)
var_95 = 0.95 * tot_var
print("95 % total variance is = ", var_95)

# creating a data field with the expalained variance
a = zip(range(0,n_components), pca.explained_variance_)
a = pd.DataFrame(a,columns=["PCA Comp", "Explained Variance"])

# Search for 95 %

print("Variance expl. with 70 components:", sum(a["Explained Variance"][0:70]))
print("Variance expl. with 75 components:", sum(a["Explained Variance"][0:75]))
print("Variance expl. with 80 components:", sum(a["Explained Variance"][0:80]))
print("Variance expl. with 85 components:", sum(a["Explained Variance"][0:85]))
print("Variance expl. with 90 components:", sum(a["Explained Variance"][0:90]))
print("Variance expl. with 95 components:", sum(a["Explained Variance"][0:95]))
print("Variance expl. with 91 components:", sum(a["Explained Variance"][0:91]))
print("Variance expl. with 92 components:", sum(a["Explained Variance"][0:92]))
print("Variance expl. with 93 components:", sum(a["Explained Variance"][0:93]))

# 92 components
pca = PCA(n_components=91, random_state = 453)
#pca = PCA(n_components=n_components)
x_r = pca.fit(x_train).transform(x_train)

############################
# determine the number of clusters - gmm
############################


S = []
bic = []
n_cluster_range = range(2,10)
for n_cluster in n_cluster_range:
    gmm = mixture.GaussianMixture(n_components = n_cluster)
    gmm.fit(x_r)
    lab = gmm.predict(x_r)
    S.append(silhouette_score(x_r,lab))
    bic.append(gmm.bic(x_r))
    
fig, (ax1, ax2) = plt.subplots(1,2, figsize= (20,10))

ax1.plot(n_cluster_range,S)
ax1.set_title('Silhouette Score')
ax1.set(xlabel= 'Number of clusters', ylabel= 'Silhouette Score')
    
ax2.plot(n_cluster_range,bic)
ax2.set_title('baysian info crit')
ax2.set(xlabel= 'Number of clusters', ylabel= 'bic')


#print("silsc = ", S)
#print("bic = ", bic)

plt.show

##########################

############################
# determine the number of clusters - kmeans
############################


k = 2
gmm = mixture.GaussianMixture(n_components=k, covariance_type='full')
gmm.fit(x_r)
predictions = gmm.predict(x_r)
#probs = gmm.predict_proba(x_r)

unique, counts = np.unique(predictions, return_counts=True)
counts = counts.reshape(1,k)

target_names = ["Cluster0", "Cluster1"]
countscldf = pd.DataFrame(counts, columns=target_names)                          
print(countscldf)
plt.figure()
plt.figure(figsize=(12,8))
colors = ['navy', 'red']
lw = 2

y_num = predictions

for color, i, target_name in zip(colors, range(0,k), target_names):
    plt.scatter(x_r[y_num == i,0], x_r[y_num ==i,1], color=color, alpha=0.8, lw=lw, label=target_name)
    
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.6)
plt.title('gmm')
plt.show()

# Add the cluster to the original dataset
df['cluster'] = predictions


df.loc[df.cluster == 0, 'cluster_Categorie'] = "Cluster 1"
df.loc[df.cluster == 1, 'cluster_Categorie'] = "Cluster 2"

############################
# Analyze of the clusters
############################

ges = []
cat1 = []
cat2 = []
quot = []
relev = []
anz1 = sum(df.cluster_Categorie == "Cluster 1")
anz2 = sum(df.cluster_Categorie == "Cluster 2")
anz = anz1+anz2

overall1 = anz1/(anz)
overall2 = anz2/(anz)

print("over all 1", overall1)
print("over all 2", overall2)


# Split the features in categories =>  Explain from which categorie the 
# people of certain feature come from.
for f in allfeatures:
    unique_vals = np.unique(df[f].apply(str))
    nr_vals = len(unique_vals)
    print("")
    print("xxxxxxx")
    print("Feature = ", f)
    for uv in unique_vals:
        u = sum(df[f].apply(str) == uv)
        u1 = sum((df[f].apply(str) == uv) & (df.cluster_Categorie == "Cluster 1"))
        u2 = sum((df[f].apply(str) == uv) & (df.cluster_Categorie == "Cluster 2"))
        abw1 = u1/u - overall1
        abw2 = u2/u - overall2
        ges.append(u)
        cat1.append(u1)
        cat2.append(u2)
        print("")
        print("Auspraegung = ", uv)
        print("Quote1 = ", u1/u, "Quote2 = ", u2/u)
        

