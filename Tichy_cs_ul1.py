# 1. Load the data
# 2. -
# 3. clean and categorize the data
# 4.a Drop the variables of set 1 and 2.
# 4.b Drop the variables of set 3 and 4.
# 5. analyze the data 



import pandas as pd
import numpy as np
import csv
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, k_means
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import mixture
from sklearn.metrics import silhouette_score


with open('head_of_columns.csv') as namen:
    namelist = csv.reader(namen, delimiter=',') 
    for zeile in namelist:
        namenliste = zeile
df = pd.read_csv('mental_health_c.csv', encoding = 'cp850', names = namenliste)


print("original data set")
print(df.shape)
print("") 


# Insert "No Answer" to empty cells
for c in df.columns:
    df[c] = df[c].fillna("No Answer")




############################
# Cleaning
############################


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





############################################
## Analyse of the  reduced feature set ##
############################################


# feature set without the features only filled for self employed or non self employed
allfeatures = ['self_employed','have_prev_emp',
               'prev_emp_prov_mh', 'prev_you_know_options_mh',
               'prev_employer_discussed_mh', 'prev_employer_offer_resources_mh',
               'prev_anonymity_protected_mh', 'prev_discussing_is_negativ_mh',
               'prev_discussing_is_negative_ph', 'prev_like_to_discuss_coworker_mh',
               'prev_like_to_discuss_supervisor_mh', 'prev_employer_ph_to_mh',
               'prev_heard_of_neg_cons_mh', 'ph_issue_in_job_interview', 'mh_issue_in_job_interview',
               'mh_issue_neg_career', 'neg_view_coworker',
               'mh_share_fam', 'neg_respons_on_mh', 'mh_in_family', 'own_mh_in_past', 'own_mh_current', 'own_diag_mh', 
               'own_treatment_mh', 'agec', 'gender', 'countryc', 
               'work_countryc', 'remote']


   
# second overview to the different answer. Only for the feature set 0. 
for f in allfeatures:
    unique_vals = np.unique(df[f].apply(str))
    nr_vals = len(unique_vals)
    if nr_vals < 10:
        print ("number of values for the feature {}: {} -- {}".format(f, nr_vals, unique_vals))
    else:
        print ("number of values for the feature {}: {}".format(f, nr_vals))




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

print("Variance expl. with 40 components:", sum(a["Explained Variance"][0:40]))
print("Variance expl. with 45 components:", sum(a["Explained Variance"][0:45]))
print("Variance expl. with 50 components:", sum(a["Explained Variance"][0:50]))
print("Variance expl. with 55 components:", sum(a["Explained Variance"][0:55]))
print("Variance expl. with 56 components:", sum(a["Explained Variance"][0:56]))
print("Variance expl. with 57 components:", sum(a["Explained Variance"][0:57]))
print("Variance expl. with 58 components:", sum(a["Explained Variance"][0:58]))
print("Variance expl. with 59 components:", sum(a["Explained Variance"][0:59]))
print("Variance expl. with 60 components:", sum(a["Explained Variance"][0:60]))


# 58 components
pca = PCA(n_components=58, random_state = 453)
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
    
fig, (ax1, ax2) = plt.subplots(1,2, figsize= (25,10))

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


S = []
inertia = []
n_cluster_range = range(2,10)
for f in n_cluster_range:
    #kmeans = KMeans(n_clusters=f, random_state=2)
    kmeans = KMeans(n_clusters=f)
    kmeans = kmeans.fit(x_r)
    pred = kmeans.predict(x_r)
    S.append(silhouette_score(x_r,pred))
    u = kmeans.inertia_
    inertia.append(u)
    print("The inertia for: ",f, "Clusters is", u)
    
fig, (ax1,ax2) = plt.subplots(1,2)

ax1.plot(n_cluster_range,S)
ax1.set_title('Silhouette Score')
ax1.set(xlabel= 'Number of clusters', ylabel= 'Silhouette Score')

# Creating the inertia plot: (elbow method) other metric?
xx = np.arange(len(n_cluster_range))  
ax2.plot(xx, inertia)
ax2.set(xlabel='Number of clusters', ylabel = 'inertia plot per k')
ax2.set_title('inertia score')

plt.show


############################
# Analyze 
############################

k = 3
gmm = mixture.GaussianMixture(n_components=k, covariance_type='full')
gmm.fit(x_r)
predictions = gmm.predict(x_r)
#probs = gmm.predict_proba(x_r)

unique, counts = np.unique(predictions, return_counts=True)
counts = counts.reshape(1,k)

target_names = ["Cluster0", "Cluster1", "Cluster2"]
countscldf = pd.DataFrame(counts, columns=target_names)                          
print(countscldf)
plt.figure()
plt.figure(figsize=(12,8))
colors = ['navy', 'red', 'yellow']
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
df.loc[df.cluster == 2, 'cluster_Categorie'] = "Cluster 3"



############################
# Analyze of the clusters
############################
    
ges = []
cat1 = []
cat2 = []
cat3 = []
quot = []
relev = []
anz1 = sum(df.cluster_Categorie == "Cluster 1")
anz2 = sum(df.cluster_Categorie == "Cluster 2")
anz3 = sum(df.cluster_Categorie == "Cluster 3")
anz = anz1+anz2+anz3

overall1 = anz1/(anz1 + anz2 + anz3)
overall2 = anz2/(anz1 + anz2 + anz3)
overall3 = anz3/(anz1 + anz2 + anz3)

print("over all 1", overall1)
print("over all 2", overall2)
print("over all 3", overall3)


############################################
## Analyse of the minimum feature set ##
############################################


# Minimal feature set. Anwered by all persons.
allfeatures = ['self_employed','have_prev_emp',
       'ph_issue_in_job_interview',
       'mh_issue_in_job_interview',
        'mh_issue_neg_career', 'neg_view_coworker',
       'mh_share_fam', 'neg_respons_on_mh', 'mh_in_family', 'own_mh_in_past', 'own_mh_current', 'own_diag_mh', 
       'own_treatment_mh', 'agec', 'gender', 'countryc', 
       'work_countryc', 'remote']


   
# second overview to the different answer. Only for the feature set 0. 
for f in allfeatures:
    unique_vals = np.unique(df[f].apply(str))
    nr_vals = len(unique_vals)
    if nr_vals < 10:
        print ("number of values for the feature {}: {} -- {}".format(f, nr_vals, unique_vals))
    else:
        print ("number of values for the feature {}: {}".format(f, nr_vals))




#one-hot coding for the categorial features
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

print("Variance expl. with 30 components:", sum(a["Explained Variance"][0:30]))
print("Variance expl. with 31 components:", sum(a["Explained Variance"][0:31]))
print("Variance expl. with 32 components:", sum(a["Explained Variance"][0:32]))
print("Variance expl. with 55 components:", sum(a["Explained Variance"][0:55]))
print("Variance expl. with 60 components:", sum(a["Explained Variance"][0:60]))
print("Variance expl. with 57 components:", sum(a["Explained Variance"][0:57]))
print("Variance expl. with 58 components:", sum(a["Explained Variance"][0:58]))
print("Variance expl. with 92 components:", sum(a["Explained Variance"][0:92]))
print("Variance expl. with 93 components:", sum(a["Explained Variance"][0:93]))


# 32 components
pca = PCA(n_components=32, random_state = 453)
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
    
fig, (ax1, ax2) = plt.subplots(1,2, figsize= (25,10))

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



S = []
inertia = []
n_cluster_range = range(2,10)
for f in n_cluster_range:
    #kmeans = KMeans(n_clusters=f, random_state=2)
    kmeans = KMeans(n_clusters=f)
    kmeans = kmeans.fit(x_r)
    pred = kmeans.predict(x_r)
    S.append(silhouette_score(x_r,pred))
    u = kmeans.inertia_
    inertia.append(u)
    print("The inertia for: ",f, "Clusters is", u)
    
fig, (ax1,ax2) = plt.subplots(1,2)

ax1.plot(n_cluster_range,S)
ax1.set_title('Silhouette Score')
ax1.set(xlabel= 'Number of clusters', ylabel= 'Silhouette Score')

# Creating the inertia plot: (elbow method) other metric?
xx = np.arange(len(n_cluster_range))  
ax2.plot(xx, inertia)
ax2.set(xlabel='Number of clusters', ylabel = 'inertia plot per k')
ax2.set_title('inertia score')

plt.show

## Analyse ##
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
plt.figure(figsize=(25,10))
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


# Auswertung
    
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

#print("quot = ", quot)



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
  


