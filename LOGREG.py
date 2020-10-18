#Chargement des librairie nécessaires
print ("Chargement des librairies")
import argparse
import pandas as pd
import numpy as np
import statsmodels.api
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#[0] passage en arguments en ligne de commande le nom du fichier
# et le pourcentage de data utilisé pour l'Entrainement
# python3 LOGREG.py -d data.csv -t 80 pour ouvrir le fichier data.csv et utiliser 80% pour l'entrainement

print ("Lecture du fichier csv contenant les données ")
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required = True, help = "Chemin vers le fichier csv contenant les data")
ap.add_argument("-t", "--train", required = True, help = "Pourcentage des données utilisées pour l'entrainement")
args = vars(ap.parse_args())
#df = pd.read_csv("/Users/francois/desktop/PythonCode/Quantmetry/data_v1.csv", low_memory=False)
df = pd.read_csv(args["data"], low_memory=False)
pcttrain = float(args["train"])
if pcttrain > 1:
    pcttrain = pcttrain/100
pcttest = 1 - pcttrain
print (str(pcttrain)+" des donnée utilisée pour l'entrainement "+str(pcttest)+ " des données pour le test .")

#[1] Préparation du dataframe avec profilage des données
#[1.1] Suppression des colonnes inutiles
print ("Préparation des données")

df=df.drop(df.columns[[1,2,3]],axis='columns')

#[1.2] Rejet des lignes commportant des cellules vides
print ("Rejet des lignes comportant des cellules vides")
df.dropna(inplace=True)

#[1.3]rejet des lignes comportant des valeurs incohérentes
print ("Rejet des lignes comportant des valeur incohérentes")
df = df[df.age > 0]
df = df[df.exp > 0]

#[1.4] transformation des variable catégorielle en bouleen
print ("Tranformation des variables catégorielles en bouléen")
df=pd.get_dummies(df, columns=['sexe','diplome','specialite','dispo'],drop_first=True)

#[2]Constitution du jeu d'entrainement et de test
#[2.1] Séparation du jeu de test (20% des données) selon une répartition aléatoire
print ("Constitution des jeux d'entrainement et de test")
X_train, X_test, y_train, y_test = train_test_split (df.drop('embauche',axis=1),
    df['embauche'], test_size=pcttest, random_state=42)

#[3]Elaboration du modèle de régression Logistique
print ("Elaboration du modèle de régression logistique")
logreg = LogisticRegression()

#[3.1] Entrainement
print ("Entrainement")
logreg.fit(X_train,y_train)

#[3.2] Prédictions
print ("Prédictions")
predictions = logreg.predict(X_test)

#[3.3] Evaluation
print ("Evaluation")
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

#[3.4] Résumé du modèle logit de statsmodels
print ("Résumé du modèle logit de statsmodels")
import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary2())

#[3.5]Calcul de la precision de la régression logistique
y_pred = logreg.predict(X_test)
print ('Accuracy of LogisticRegression classifier on test set:\n')
print ('{:.2f}'.format(logreg.score(X_test,y_test)))

#[3.6]Autre méthode pour caculer les matrice de confusion
print ('Matrice de confusion\n')
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix (y_test,y_pred)
print (confusion_matrix)
