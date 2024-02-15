# 1. Load the data
# 2. clean and categorize the data
# 3. Split the dataset in self employed and non self employed
# 4. a Drop the variables of set 3 and 4.
# 4. b define sets of variables with set 0 and set 1 (non self employed)
#    and variables with set 0 and set 2 (self employed)
#    ! choosen by the variable dfself, used in the code below!!! (not the best style for sure.) 
# 5. analyze the data 



####################
# Analyse for self employed - non self employed group


import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
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


df['mh'] = "No"
df.loc[df.own_mh_in_past == "Yes", 'mh'] = "Yes"
df.loc[df.own_mh_current == "Yes", 'mh'] = "Yes"



############################
# Selection of the dataset
############################

df_self = df.loc[df.self_employed == 1]

print("self employed data set")
print(df_self.shape)
print("") 

df_nonself = df.loc[df.self_employed == 0]

print("non self employed data set")
print(df_nonself.shape)
print("") 



############################
# Selection of the features
############################


# tech_role', 'state','work_state', 'position', 'why_ph_issue_in_job_int', 'why_mh_issue_in_job_int',
# 'which_own_diag_mh', 'sure_kind_own_mh', 'maybe_kind_own_mh',
allfeaturesself = ['self_employed', 'med_cov_with_mh', 'know_res_seek_mh',
       'diag_or_treat_mh', 'bus_cont_mh_is_neg_input',
       'reveal_own_mh_to_coworker', 'have_rev_mh_to_coworker_believe_neg',
       'prod_eff_by_mh', 'perc_work_time_eff_by_mh', 'have_prev_emp',
       'ph_issue_in_job_interview',
       'mh_issue_in_job_interview', 'mh',
        'mh_issue_neg_career', 'neg_view_coworker',
       'mh_share_fam', 'neg_respons_on_mh', 'mh_in_family', 'own_diag_mh', 
       'own_treatment_mh', 'agec', 'gender', 'countryc', 
       'work_countryc', 'remote']

#'own_mh_in_past', 'own_mh_current',

allfeaturesnonself = ['self_employed', 'number_employees', 'tech_organisation',
       'employer_provide_mh', 'you_know_options_mh', 'employer_discussed_mh',
       'employer_offer_resources_mh', 'anonymity_protected_mh',
       'able_leave_for_mh', 'discussing_is_negativ_mh',
       'discussing_is_negative_ph', 'like_to_discuss_coworker_mh',
       'like_to_discuss_supervisor_mh', 'employer_ph_to_mh',
       'heard_of_neg_cons_mh', 'have_prev_emp',
       'ph_issue_in_job_interview',
       'mh_issue_in_job_interview',
        'mh_issue_neg_career', 'neg_view_coworker',
       'mh_share_fam', 'neg_respons_on_mh', 'mh_in_family', 'own_mh_in_past', 'own_mh_current', 'own_diag_mh', 
       'own_treatment_mh', 'agec', 'gender', 'countryc', 
       'work_countryc', 'remote']


#########
### For the self employed     => dfself = 1
### For the non self employed => dfself = 0

#########



############################
# Coding of the features
############################

#For the self employed: dfself = 1
# For the non self employed: dfself = 0
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dfself = 0
if dfself == 1:
    aktdf = df_self
    aktfeature = allfeaturesself
    raw_aktdf = df_self[allfeaturesself]
elif dfself  == 0:
    aktdf = df_nonself
    aktfeature = allfeaturesnonself
    raw_aktdf = df_nonself[allfeaturesnonself]

    
new_raw_aktdf = pd.get_dummies(data = raw_aktdf, columns = aktfeature)
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
print("Variance expl. with 42 components:", sum(a["Explained Variance"][0:42]))
print("Variance expl. with 49 components:", sum(a["Explained Variance"][0:49]))
print("Variance expl. with 50 components:", sum(a["Explained Variance"][0:50]))
print("Variance expl. with 51 components:", sum(a["Explained Variance"][0:51]))
print("Variance expl. with 55 components:", sum(a["Explained Variance"][0:55]))
print("Variance expl. with 56 components:", sum(a["Explained Variance"][0:56]))
print("Variance expl. with 57 components:", sum(a["Explained Variance"][0:57]))
print("Variance expl. with 58 components:", sum(a["Explained Variance"][0:58]))


# 57 components
pca = PCA(n_components=57, random_state = 453)
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
aktdf['cluster'] = predictions
aktdf.loc[aktdf.cluster == 0, 'cluster_Categorie'] = "Cluster 1"
aktdf.loc[aktdf.cluster == 1, 'cluster_Categorie'] = "Cluster 2"


############################
# Analyze of the clusters
############################

    
ges = []
cat1 = []
cat2 = []
quot = []
relev = []
anz1 = sum(aktdf.cluster_Categorie == "Cluster 1")
anz2 = sum(aktdf.cluster_Categorie == "Cluster 2")
anz = anz1+anz2

overall1 = anz1/(anz)
overall2 = anz2/(anz)

print("over all 1", overall1)
print("over all 2", overall2)

#print("quot = ", quot)



# Split the features in categories =>  Explain from which categorie the 
# people of certain feature come from.
for f in aktfeature:
    unique_vals = np.unique(df[f].apply(str))
    nr_vals = len(unique_vals)
    print("")
    print("xxxxxxx")
    print("Feature = ", f)
    for uv in unique_vals:
        u = sum(aktdf[f].apply(str) == uv)
        if (u != 0):
            u1 = sum((aktdf[f].apply(str) == uv) & (aktdf.cluster_Categorie == "Cluster 1"))
            u2 = sum((aktdf[f].apply(str) == uv) & (aktdf.cluster_Categorie == "Cluster 2"))
            abw1 = u1/u - overall1
            abw2 = u2/u - overall2
            ges.append(u)
            cat1.append(u1)
            cat2.append(u2)
            print("")
            print("Auspraegung = ", uv)
            print("Quote1 = ", u1/u, "Quote2 = ", u2/u)
        else:
            print("")
            print("Auspraegung = ", uv)
            print("Nicht vorhanden")    
  

# Characterise the categories: That means which answers have the people of the
# categorie given.

print('++++++++++++++++++++++++++++++++++++++++++')
print('++++++++++++++++++++++++++++++++++++++++++')
print('++++++++++++++++++++++++++++++++++++++++++')

for f in aktfeature:
    unique_vals = np.unique(aktdf[f].apply(str))
    nr_vals = len(unique_vals)
    print("")
    print("xxxxxxx")
    print("Feature = ", f)
    ges = []
    cl1 = []
    cl2 = []
    ans = []
    for uv in unique_vals:
        u = sum(aktdf[f].apply(str) == uv)
        u1 = sum((aktdf[f].apply(str) == uv) & (aktdf.cluster_Categorie == "Cluster 1"))
        u2 = sum((aktdf[f].apply(str) == uv) & (aktdf.cluster_Categorie == "Cluster 2"))
        u3 = sum((aktdf[f].apply(str) == uv) & (aktdf.cluster_Categorie == "Cluster 3"))
        ges.append(u/anz)
        cl1.append(u1/anz1)
        cl2.append(u2/anz2)
        ans.append(uv)
    print("")
    print("Answer", ans)
    print("gesamt = ", ges)
    print("Categorie 1 = ", cl1)
    print("Categorie 2 = ", cl2)
    
