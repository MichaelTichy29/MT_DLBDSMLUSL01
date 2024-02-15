#analyse of the text feature

import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import mixture
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, k_means

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




# NOT ENOUGHT structure: why_mh_issue_in_job_int, why_ph_issue_in_job_int 



################
###### 'which_own_diag_mh' as text var ######
######     boW #####
##################

# Variables of text "which_own_diag_mh"
text = df["which_own_diag_mh"]

vectorizer = CountVectorizer(lowercase = False, stop_words='english')
BoW = vectorizer.fit_transform(text)

#print(vectorizer.get_feature_names())
inhalt = vectorizer.get_feature_names()
df_bow = BoW.toarray()
#print(df_bow)
###print(df_bow.shape)
#print("inhalt = ", inhalt)
df_which_own_diag_mh = pd.DataFrame(np.array(df_bow), columns=inhalt)



################
######     frequent used words 
##################
text_var = []
for f in inhalt:
    a = sum(df_which_own_diag_mh[f])
    if a >= 50:
        #print("word = ", f, " -- Number = ", a)
        text_var.append(str(f))

# to copy the relevant words manual in the list text_var_new       
#print("text_var (which own diag mh) = ", text_var)

text_var_new =  ['Anxiety', 'Attention', 'Bipolar', 'Deficit', 'Depression', 'Disorder', 'Generalized', 'Hyperactivity', 'Mood', 'Phobia', 'Post', 'Social', 'Stress', 'traumatic']


z = df.shape
for f in text_var_new:
    df[f] = 0
    for k in range(1,z[0]):
        if f in df.iloc[k]['which_own_diag_mh']:
            df.at[k,f] = 1

################
###### 'sure_kind_own_mh' and 'maybe_kind_own_mh' as text var ######
######     boW #####
##################


df["own_mh"] = df["sure_kind_own_mh"].map(str) + " " + df["maybe_kind_own_mh"].map(str)

text = df["own_mh"]


vectorizer = CountVectorizer(lowercase = False, stop_words='english')
BoW = vectorizer.fit_transform(text)

#print(vectorizer.get_feature_names())
inhalt = vectorizer.get_feature_names()
df_bow = BoW.toarray()
#print(df_bow)
###print(df_bow.shape)
#print("inhalt = ", inhalt)
df_own_mh = pd.DataFrame(np.array(df_bow), columns=inhalt)


################
######     frequent used words 
##################

text_var = []
for f in inhalt:
    a = sum(df_own_mh[f])
    if a >= 50:
        #print("word = ", f, " -- Number = ", a)
        text_var.append(str(f))

# to copy the relevant words manual in the list text_var_new       
#print("text_var (own mh) = ", text_var)


text_var_new0 =  ['Addictive', 'Antisocial', 'Anxiety', 'Attention', 'Bipolar', 'Borderline', 'Compulsive', 'Depression', 'Disorder', 'Generalized', 'Hyperactivity', 'Mood', 'Obsessive', 'Paranoid', 'Personality', 'Phobia', 'Social', 'Stress', 'traumatic']

text_var_new_own = []

# Add "_own" to the variables to be able to separete them from the which own diag mh
for word in text_var_new0:
    wordnew = "own_" + word
    text_var_new_own.append(wordnew)


z = df.shape
for f in text_var_new_own:
    df[f] = 0
    for k in range(1,z[0]):
        if f[4:] in df.iloc[k]['own_mh']:
            df.at[k,f] = 1

#print("own: text_var new", text_var_new)


####################
# Build the features set with the minimal features and the features of the text.
####################

featuresmin = ['self_employed',  'have_prev_emp',
       'ph_issue_in_job_interview',
       'mh_issue_in_job_interview',
       'mh_issue_neg_career', 'neg_view_coworker',
       'mh_share_fam', 'neg_respons_on_mh', 
       'mh_in_family', 'own_mh_in_past', 'own_mh_current','own_diag_mh', 
       'own_treatment_mh', 'agec', 'gender', 'countryc', 
       'work_countryc', 'remote']

features1 = ['self_employed',  'have_prev_emp',
       'ph_issue_in_job_interview',
       'mh_issue_in_job_interview',
       'mh_issue_neg_career', 'neg_view_coworker',
       'mh_share_fam', 'neg_respons_on_mh', 
       'mh_in_family', 'own_mh_in_past', 'own_mh_current','own_diag_mh', 
       'own_treatment_mh', 'agec', 'gender', 'countryc', 
       'work_countryc', 'remote']
features2 = []

features = featuresmin

for word in text_var_new:
    features.append(word)
    features2.append(word)

for word in text_var_new_own:
    features.append(word)
    features2.append(word)
    
# Build the dateframe with the rows, foir which "own_mh_current" or own_mh_past = Yes or Maybe 
# and the rows for which "own_diag_mh" = Yes 


################
######     Selection of dataset (only the rows with text fields) 
##################
    

df['textrel'] = 0
df.loc[df.own_mh_in_past == "Yes", 'textrel'] = 1
df.loc[df.own_mh_current == "Yes", 'textrel'] = 1
df.loc[df.own_mh_in_past == "Maybe", 'textrel'] = 1
df.loc[df.own_mh_current == "Maybe", 'textrel'] = 1
df.loc[df.own_diag_mh == "Yes", 'textrel'] = 1


df_text = df.loc[df.textrel == 1]

print("Date set with text rel")
print(df_text.shape)
print("") 
 

raw_text = df_text[features]




############################
# Coding of the features
############################
  
new_raw_text = pd.get_dummies(data = raw_text, columns = features)
print("shape of dummy data is")
print(new_raw_text.shape)


######################
# Analyse of the dataset
######################
x_train = new_raw_text.values



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


print("Variance expl. with 38 components:", sum(a["Explained Variance"][0:38]))
print("Variance expl. with 39 components:", sum(a["Explained Variance"][0:39]))
print("Variance expl. with 40 components:", sum(a["Explained Variance"][0:40]))
print("Variance expl. with 41 components:", sum(a["Explained Variance"][0:41]))
print("Variance expl. with 71 components:", sum(a["Explained Variance"][0:71]))
print("Variance expl. with 72 components:", sum(a["Explained Variance"][0:72]))
print("Variance expl. with 73 components:", sum(a["Explained Variance"][0:73]))
print("Variance expl. with 74 components:", sum(a["Explained Variance"][0:74]))
print("Variance expl. with 75 components:", sum(a["Explained Variance"][0:75]))



# 39 components as as result 
pca = PCA(n_components=39, random_state = 453)
#pca = PCA(n_components=n_components)
x_r = pca.fit(x_train).transform(x_train)



############################
# determine the number of clusters - gmm
############################


S = []
bic = []
n_cluster_range = range(2,10)
for n_cluster in n_cluster_range:
    gmm = mixture.GaussianMixture(n_components = n_cluster, random_state = 453)
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
# determine the number of clusters - k-means
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
#ax2.plot(xx, inertia)
ax2.plot(n_cluster_range, inertia)
ax2.set(xlabel='Number of clusters', ylabel = 'inertia plot per k')
ax2.set_title('inertia score')

plt.show


################ Analyse for k = 5 #############
k = 5
gmm = mixture.GaussianMixture(n_components=k, covariance_type='full', random_state = 453)
gmm.fit(x_r)
predictions = gmm.predict(x_r)
#probs = gmm.predict_proba(x_r)

unique, counts = np.unique(predictions, return_counts=True)
counts = counts.reshape(1,k)

target_names = ["Cluster0", "Cluster1", "Cluster2", "Cluster3", "Cluster4"]
countscldf = pd.DataFrame(counts, columns=target_names)                          
print(countscldf)
plt.figure()
plt.figure(figsize=(25,10))
colors = ['navy', 'red', 'blue', 'orange', 'black']
lw = 2

y_num = predictions

for color, i, target_name in zip(colors, range(0,k), target_names):
    plt.scatter(x_r[y_num == i,0], x_r[y_num ==i,1], color=color, alpha=0.8, lw=lw, label=target_name)
    
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.6)
plt.title('gmm')
plt.show()



# Add the cluster to the original dataset
df_text['cluster'] = predictions
df_text.loc[df_text.cluster == 0, 'cluster_Categorie'] = "Cluster 1"
df_text.loc[df_text.cluster == 1, 'cluster_Categorie'] = "Cluster 2"
df_text.loc[df_text.cluster == 2, 'cluster_Categorie'] = "Cluster 3"
df_text.loc[df_text.cluster == 3, 'cluster_Categorie'] = "Cluster 4"
df_text.loc[df_text.cluster == 4, 'cluster_Categorie'] = "Cluster 5"


# Auswertung
anz1 = sum(df_text.cluster_Categorie == "Cluster 1")
anz2 = sum(df_text.cluster_Categorie == "Cluster 2")
anz3 = sum(df_text.cluster_Categorie == "Cluster 3")
anz4 = sum(df_text.cluster_Categorie == "Cluster 4")
anz5 = sum(df_text.cluster_Categorie == "Cluster 5")

anz = anz1+anz2 + anz3 + anz4 + anz5




############################
# Analyze of the clusters
############################

print('++++++++++++++++++++++++++++++++++++++++++')
print('++++++++++++++++++++++++++++++++++++++++++')
print('++++++++++++++++++++++++++++++++++++++++++')

for f in features2:
    unique_vals = np.unique(df_text[f].apply(str))
    nr_vals = len(unique_vals)
    print("")
    print("xxxxxxx")
    print("Feature = ", f)
    ges = []
    cl1 = []
    cl2 = []
    cl3 = []
    cl4 = []
    cl5 = []
    
    ans = []
    for uv in unique_vals:
        u = sum(df_text[f].apply(str) == uv)
        u1 = sum((df_text[f].apply(str) == uv) & (df_text.cluster_Categorie == "Cluster 1"))
        u2 = sum((df_text[f].apply(str) == uv) & (df_text.cluster_Categorie == "Cluster 2"))
        u3 = sum((df_text[f].apply(str) == uv) & (df_text.cluster_Categorie == "Cluster 3"))
        u4 = sum((df_text[f].apply(str) == uv) & (df_text.cluster_Categorie == "Cluster 4"))
        u5 = sum((df_text[f].apply(str) == uv) & (df_text.cluster_Categorie == "Cluster 5"))
        ges.append(u/anz)
        cl1.append(u1/anz1)
        cl2.append(u2/anz2)
        cl3.append(u3/anz3)
        cl4.append(u4/anz4)
        cl5.append(u5/anz5)
        ans.append(uv)
    print("")
    print("Answer", ans)
    print("gesamt = ", ges)
    print("Categorie 1 = ", cl1)
    print("Categorie 2 = ", cl2)
    print("Categorie 3 = ", cl3)
    print("Categorie 4 = ", cl4)
    print("Categorie 5 = ", cl5)
















"""
###########
### idea for two word expressions. Droped at the moment. 
############
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2), lowercase=False,
stop_words='english')
Bo2G = vectorizer2.fit_transform(text)

inhalt = vectorizer2.get_feature_names()
print(inhalt)


#print(Bo2G.toarray())
df_2bow = Bo2G.toarray()
print(df_2bow.shape)

df_which_own_diag_mh2 = pd.DataFrame(np.array(df_2bow), columns=inhalt)

text_var = []

for f in inhalt:
    a = sum(df_which_own_diag_mh2[f])
    if a >= 50:
        #print("word = ", f, "Number = ", sum(df_which_own_diag_mh[f]))
        print("word = ", f, " -- Number = ", a)        text_var.append(f
"""    
        
