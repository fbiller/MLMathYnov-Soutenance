import argparse
from pandas_profiling import ProfileReport
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
import matplotlib
from matplotlib import pyplot
import scipy.stats as ss
import statsmodels.api
import researchpy as rp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

#[0] passage en arguments en ligne de commande le nom du fichier
# et le pourcentage de data utilisé pour l'Entrainement
# python3 STATDES.py -d data.csv
print ("Lecture du fichier csv contenant les données ")
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required = True, help = "Chemin vers le fichier csv contenant les data")
args = vars(ap.parse_args())
df = pd.read_csv(args["data"], low_memory=False)
#[1] profilage des données
#[1.1]detail report on the data included in data_v1.read_csv

print ("#[1]============ Profilage du jeux de données ================")
prof=ProfileReport(df)
prof.to_file(output_file="rapport_Data_Set.html")

#[2.1] Rejet des lignes commportant des cellules vides
print ("Rejet des lignes comportant des cellules vides")
df.dropna(inplace=True)

#[2.2] retirons ces lignes avec des valeurs incohérentes,
#plutôt ne conservons que celles qui sont cohérentes
df = df[df.age > 0]
df = df[df.exp > 0]

# assigne le format de date à la colonne 'date'
df["date"] = pd.to_datetime(df["date"])

# façon de reduire de la taille mémoire pour le jeu de donnée
df["diplome"] = pd.Categorical(df["diplome"])
df["sexe"] = pd.Categorical(df["sexe"])
df["specialite"] = pd.Categorical(df["specialite"])
df["dispo"] = pd.Categorical(df["dispo"])

#k=df[["exp",'note']]
#pyplot.scatter(k[0],k[1],c='red',marker='*',edgecolors='blue')
#print ("Pearson",k.corr(method='pearson'))
#print ("Kendall",k.corr(method="kendall"))

#[3]table de contingence sexe/specialite et test du CHI2
table = pd.crosstab (df["sexe"],df["specialite"])
print ("#[3]===========TEST CHI2 pour les variables sexe/specialite\n",)
resultats_test = chi2_contingency(table)
p_valeur = resultats_test[1]
print("statistique de test :",resultats_test[0])
print("p_valeur :",p_valeur)
print("degré de liberté :",resultats_test[2])

if p_valeur < 0.1 :
    print ("Il existe une dépendances statistique entre les variables sexe & specialite")
    print ("car p-valeur est intérieur au niveau de signification de 5%",)

#[4] Calcul du V de cramer corrigé
def cramers_corrected_stat (confusion_matrix):
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr=max(0,phi2-((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

print ("#[4] cramers_corrected_stat=",cramers_corrected_stat(table))
print ("la relation entre ces 2 variable est modeste, car le calcul du V de cramer corrigé ci-dessus")
print ("est plus proche de zéro (relation faible) que de 0,9 (relation très forte)")
print ("\n")

#[6] One Way anova with python
#print("\n\n#[6]==========librairie researchpy===============")
#[6.1] Calcul et affiche le calcul de la table Anova du salaire pour toutes les catégories de cheveux
print ("#[6.1] ANOVA ===salaire seul===\n",rp.summary_cont(df['salaire']))
#[6.2] Calcul et affiche le calcul de la table Anova du salaire par couleur de cheveux
print ("#[6.2] ANOVA ===salaire groupé par type de cheveux===\n",
       rp.summary_cont(df['salaire'].groupby(df['cheveux'])))
#[6.3] Calcul et affiche F-statistic and p-value du test
foneway=ss.f_oneway (df['salaire'][df['cheveux']  == 'chatain'],
             df['salaire'][df['cheveux'] == 'brun'],
             df['salaire'][df['cheveux'] == 'blond'],
             df['salaire'][df['cheveux'] == 'roux'])
print ("\n#[6.3]====foneway====\n",foneway)



#[6.4] Calcul et affiche le levene test de l'homogenéité de la variance
statlevene=ss.levene (df['salaire'][df['cheveux']  == 'chatain'],
             df['salaire'][df['cheveux'] == 'brun'],
             df['salaire'][df['cheveux'] == 'blond'],
             df['salaire'][df['cheveux'] == 'roux'])
print ("\n#[6.4]====statlevene=====\n",statlevene)
print ("\n")

#[6.5] Calcul et affichage de la représentation graphique "moustache"
fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)
ax.set_title("Box Plot du Salaire selon la couleur des cheveux", fontsize= 20)
ax.set

data = [df['salaire'][df['cheveux']  == 'chatain'],
             df['salaire'][df['cheveux'] == 'brun'],
             df['salaire'][df['cheveux'] == 'blond'],
             df['salaire'][df['cheveux'] == 'roux']]

ax.boxplot(data,
           labels= ['chatain', 'brun', 'blond', 'roux'],
           showmeans= True)

plt.xlabel("Couleur des cheuveux")
plt.ylabel("Salaire")
plt.show(block=False)
print("Pour revenir au programe fermer la fenetre d'affichage du graphique")
plt.show()
print("Fin d'affichage du graphique")

#[6.6] Analyse de la variance pour les variables courleur des cheuveux et salaire
results = statsmodels.formula.api.ols('salaire ~ cheveux',data=df).fit()
table_anova = statsmodels.api.stats.anova_lm(results)
print("#[6.6]============table_anova==============\n",table_anova)


#[7]Calcul et affichage du Test de Pearson
TPearson=pd.DataFrame(pearsonr(df['exp'],df['note']),
             index = ['pearson_coeff','p-value'],
             columns = ['resultats_test'])
print ("#[7] ======== Test de Pearson pour les variable ''exp'' et ''note'' \n",TPearson)
