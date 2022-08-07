# -*- coding: utf-8 -*-

import os
os.chdir("D:/_Travaux/university/Cours_Universite/Supports_de_cours/Informatique/Python/Slides/exemples/K")

#utilisation de la librairie Pandas
#spécialisée - entres autres - dans la manipulation des données
import pandas
cars = pandas.read_table("vehicules_1.txt",sep="\t",header=0,index_col=0)

#dimensions
print(cars.shape)

#nombre d'observations
n = cars.shape[0]

#nombre de variables explicatives
p = cars.shape[1] - 1

#liste des colonnes et leurs types
print(cars.dtypes)

#liste des modèles
print(cars.index)

##########################################
# Régression et inspection des résultats #
##########################################

#régression avec formule
import statsmodels.formula.api as smf

#instanciation
reg = smf.ols('conso ~ cylindree + puissance + poids', data = cars)

#membres de reg
print(dir(reg))

#lancement des calculs
res = reg.fit()

#liste des membres
print(dir(res))

#résultats détaillés
print(res.summary())

#paramètres estimés
print(res.params)

#le R2
print(res.rsquared)

#calcul manuel du F à partir des carrés moyens
F = res.mse_model / res.mse_resid
print(F)

#F fourni par l'objet res
print(res.fvalue)

#test avec combinaison linéaires de param.
#nullité simultanée de tous les param. sauf la constante
import numpy as np

#matrices des combinaisons à tester
R = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1]])

#on obtient le test F
print(res.f_test(R))

###############################
# Diagnostic de la régression #
###############################

### normalité des résidus

#test de normalité de Jarque-Bera
import statsmodels.api as sm
JB, JBpv,skw,kurt = sm.stats.stattools.jarque_bera(res.resid)
print(JB,JBpv,skw,kurt)

#qqpolot vs. loi normale
sm.qqplot(res.resid)

### détection des points influents

#objet pour les mesures d'influence
infl = res.get_influence()

#membres
print(dir(infl))

#leviers
print(infl.hat_matrix_diag)

#résidus standardisés
print(infl.resid_studentized_internal)

#vérifions avec la formule du cours
import numpy as np
residus = res.resid.as_matrix()
leviers = infl.hat_matrix_diag #leviers
sigma_err = np.sqrt(res.scale) #ec.type.erreur
res_stds = residus/(sigma_err*np.sqrt(1.0-leviers))
print(res_stds)

#résidus studentisés
print(infl.resid_studentized_external)

#vérifions avec la formule
res_studs = res_stds*np.sqrt((n-p-2)/(n-p-1-res_stds**2))
print(res_studs)

#graphique des influences()
sm.graphics.influence_plot(res)

#détection basée sur le levier et le résidu studentisé

#seuil levier
seuil_levier = 2*(p+1)/n
print(seuil_levier)

#identification
atyp_levier = leviers > seuil_levier
print(atyp_levier)

#quels véhicules ?
print(cars.index[atyp_levier],leviers[atyp_levier])

#seuil résidus studentisés
import scipy
seuil_stud = scipy.stats.t.ppf(0.975,df=n-p-2)
print(seuil_stud)
#détection - val. abs > seuil
atyp_stud = np.abs(res_studs) > seuil_stud
#lequels ?
print(cars.index[atyp_stud],res_studs[atyp_stud])

#problématique avec un des deux critères
pbm_infl = np.logical_or(atyp_levier,atyp_stud)
print(cars.index[pbm_infl])

#présentation sous forme de tableau
print(infl.summary_frame().filter(["hat_diag","student_resid","dffits","cooks_d"]))

###############################
# Détection de la colinéarité #
###############################

#liste des var. exogènes-matrice format numy
cars_exog = cars[['cylindree','puissance','poids']].as_matrix()

#matrice des corrélations avec scipy
import scipy
mc = scipy.corrcoef(cars_exog,rowvar=0)
print(mc)

#règle de Klein
mc2 = mc**2
print(mc2)

#critère VIF
vif = np.linalg.inv(mc)
print(vif)


###########################################
# Prédiction ponctuelle et par intervalle #
###########################################

#chargement du second fichier de données
cars2 = pandas.read_table("vehicules_2.txt",sep="\t",header=0,index_col=0)

#nombre d'obs. à prédire
n_pred = cars2.shape[0]

#liste des modèles
print(cars2)

#exogènes du fichier
cars2_exog = cars2[['cylindree','puissance','poids']]

#ajouter une constante pour que le produit matriciel fonctionne
cars2_exog = sm.add_constant(cars2_exog)
print(cars2_exog)

#réaliser la prédiction ponctuelle - reg est l'objet régression
pred_conso = reg.predict(res.params,cars2_exog)
print(pred_conso)

#confrontation obs. vs. pred.
import matplotlib.pyplot as plt
plt.scatter(cars2['conso'],pred_conso)
plt.plot(np.arange(5,23),np.arange(5,23))
plt.xlabel('Valeurs Observées')
plt.ylabel('Valeurs Prédites')
plt.xlim(5,22)
plt.ylim(5,22)
plt.show()

#récupération de la matrice (X'X)^-1
inv_xtx = reg.normalized_cov_params
print(inv_xtx)

#transformation en type matrice des éxogènes à prédire
X_pred = cars2_exog.as_matrix()

#variance de l'erreur de prédiction
#initialisation
var_err = np.zeros((n_pred,))
#pour chaque individu à traiter
for i in range(n_pred):
    #description de l'indiv.
    tmp = X_pred[i,:]
    #produit matriciel
    pm = np.dot(np.dot(tmp,inv_xtx),np.transpose(tmp))
    #variance de l'erreur
    var_err[i] = res.scale * (1 + pm)
#
print(var_err)    
    
#quantile de la loi de Student pour un intervalle à 95%
qt = scipy.stats.t.ppf(0.975,df=n-p-1)

#borne basse
yb = pred_conso - qt * np.sqrt(var_err)
print(yb)

#borne haute
yh = pred_conso + qt * np.sqrt(var_err)
print(yh)

#matrice collectant les différentes colonnes (yb,yobs,yh)
a = np.resize(yb,new_shape=(n_pred,1))
y_obs = cars2['conso']
a = np.append(a,np.resize(y_obs,new_shape=(n_pred,1)),axis=1)
a = np.append(a,np.resize(yh,new_shape=(n_pred,1)),axis=1)
print(a)

#mettre sous forme de data frame pour commodité d'affichage
df = pandas.DataFrame(a)
df.index = cars2.index
df.columns = ['B.Basse','Y.Obs','B.Haute']
print(df)