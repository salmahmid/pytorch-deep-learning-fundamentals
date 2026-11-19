# TP1 — Mise en œuvre du Framework PyTorch

## Description

Ce travail pratique présente les bases de PyTorch, puis les met en pratique sur quatre problèmes : une régression de salaire, la classification des fleurs Iris, un perceptron multicouche sur MNIST et un réseau convolutif sur MNIST.

Le notebook du TP est `Activité pratique Bases de Pytorch.ipynb`.

## Objectifs pédagogiques

- Comprendre les tenseurs et leur manipulation.
- Calculer des gradients avec l'autograd.
- Construire un modèle avec `nn.Module` et `nn.Sequential`.
- Définir une fonction de coût et un optimiseur.
- Écrire une boucle d'apprentissage, puis distinguer `train()` et `eval()`.
- Utiliser `Dataset` et `DataLoader`.
- Entraîner une régression et une classification, puis sauvegarder et recharger un modèle.
- Appliquer cette méthode à un salaire, aux fleurs Iris et aux chiffres MNIST, avec un MLP puis un CNN.

## Environnement de travail

- Système : macOS
- Python : 3.12.10
- Jupyter Notebook, noyau `.venv (3.12.10)`
- PyTorch : 2.15.0
- torchvision : 0.29.2
- Matplotlib : 3.11.2
- scikit-learn : 1.9.1
- torchviz : 0.0.3, avec l'outil système Graphviz (`dot`)
- torcheval : 0.0.7
- torchmetrics : 1.9.0
- certifi : 2026.10.23, utilisé pour le téléchargement de MNIST

L'entraînement a été exécuté sur CPU. Cette machine ne dispose pas de CUDA, et le notebook choisit le CPU lorsque CUDA n'est pas disponible.

## Installation

Depuis la racine du projet :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision matplotlib scikit-learn ipykernel torchviz torcheval torchmetrics certifi
python -m ipykernel install --user --name tp1-pytorch-venv --display-name ".venv (3.12.10)"
```

Graphviz doit aussi être installé sur le système, car `torchviz` appelle la commande `dot` pour dessiner le graphe de calcul.

Si le téléchargement de MNIST échoue avec `CERTIFICATE_VERIFY_FAILED`, indiquer à Python le fichier de certificats de certifi avant d'ouvrir le notebook :

```bash
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
```

## Exécution

1. Ouvrir `Activité pratique Bases de Pytorch.ipynb`.
2. Sélectionner le noyau `.venv (3.12.10)`, associé à `.venv/bin/python`.
3. Exécuter les cellules dans l'ordre du notebook.

Le premier lancement de MNIST crée le dossier `data/`. L'exécution produit aussi `graphe_operations.png`, `classifier_state_dict.pt`, `mnist_mlp_state_dict.pt` et `mnist_cnn.pth`.

## Partie 1

La première partie suit le cycle de PyTorch : données, tenseurs, modèle, prédiction, perte, rétropropagation, puis mise à jour des poids.

Les notions étudiées dans le notebook sont :

- les tenseurs et leurs dimensions ;
- le choix du device, CUDA s'il existe, sinon CPU ;
- l'autograd et la vérification analytique des gradients ;
- un neurone `nn.Linear` et la visualisation du graphe avec `torchviz` ;
- les réseaux `nn.Module` et `nn.Sequential` ;
- les fonctions d'activation ReLU, tanh et sigmoid ;
- la propagation avant ;
- `MSELoss` pour la régression et `CrossEntropyLoss` pour la classification ;
- l'optimiseur Adam ;
- la boucle `zero_grad`, prédiction, perte, `backward`, `step` ;
- la différence entre `train()` et `eval()`, avec un exemple de Dropout ;
- `TensorDataset` et `DataLoader` ;
- une régression linéaire et une classification binaire, avec sauvegarde du `state_dict`.

Résultats obtenus à partir de la section 9 :

- MSE de l'exemple : 1.0
- Cross-entropy de l'exemple : 0.3691
- Boucle d'apprentissage sur `y = 2x + 1` : la perte passe de 5.9272 à l'époque 0 à 0.0552 à l'époque 400
- Régression `y = 3x + 2 + bruit` : perte finale 0.3438, poids appris 2.989, biais appris 2.030, R² = 0.972
- Classification de deux nuages de points : perte finale 0.000694, accuracy 100 %, précision, rappel et F1 égaux à 1. La matrice de confusion est diagonale, 120 exemples par classe. Le modèle rechargé donne les mêmes prédictions.

## Partie 2

### Exercice 1 — Régression du salaire

Le salaire mensuel fictif, en dirhams, dépend de l'ancienneté et du niveau d'expertise. Les 300 exemples suivent la règle `4000 + 600 × ancienneté + 1500 × expertise`, plus un bruit d'écart-type 500 DH.

Le notebook sépare 80 % d'entraînement et 20 % de test, normalise les entrées avec la moyenne et l'écart-type du seul jeu d'entraînement, et exprime les salaires en milliers de DH. Le modèle est `nn.Linear(2, 1)`, entraîné 500 époques avec `MSELoss` et SGD au taux 0.05.

Résultats :

- dimensions : `X` vaut `(300, 2)` et `y` vaut `(300, 1)` ;
- salaire moyen : 15 010.55 DH ;
- perte d'entraînement stabilisée à 0.2391 ;
- erreur absolue moyenne sur le test : 381.64 DH ;
- salaire prédit pour 7 ans d'ancienneté et une expertise de 4/5 : 14 201.94 DH, proche des 14 200 DH de la règle sans bruit ;
- coefficients retrouvés dans les unités d'origine : 592.7 DH par année, 1 490.7 DH par niveau d'expertise, et un biais de 4 090.3 DH.

### Exercice 2 — Classification Iris

Le jeu Iris contient 150 fleurs, quatre mesures et trois espèces. Le notebook utilise 80 % des données pour l'entraînement, avec stratification, puis un `StandardScaler` ajusté uniquement sur l'entraînement.

Le notebook laisse vide la largeur de la couche cachée. Elle a été fixée à 16, comme dans le `Classifier` de la partie pratique. L'architecture est donc `Linear(4, 16)`, ReLU, `Linear(16, 3)`, sans Softmax. L'entraînement dure 200 époques avec `CrossEntropyLoss` et Adam au taux 0.01, sur tout le jeu d'entraînement à chaque époque.

Résultats :

- dimensions d'entraînement : `(120, 4)` et `(120,)` ;
- perte finale : 0.0432 ;
- accuracy sur le test : 100 % ;
- le modèle sans couche cachée `nn.Linear(4, 3)`, entraîné avec les mêmes hyperparamètres, atteint 90 % ;
- la fleur de mesures `[5.1, 3.5, 1.4, 0.2]` est prédite `setosa`.

Dans le notebook, le titre de cette partie est placé après le travail et s'appelle « Exercice 3 — Classification des fleurs Iris ».

### Exercice 3 — MLP MNIST

Les images 28 × 28 sont converties en tenseurs avec `ToTensor`. Les lots d'entraînement contiennent 64 images et sont mélangés. Les lots de test en contiennent 256.

L'architecture du notebook est `Flatten`, `Linear(784, 128)`, ReLU, `Linear(128, 10)`. Elle est entraînée 5 époques avec `CrossEntropyLoss` et Adam au taux 0.001, sans Softmax.

Résultats sur les 10 000 images de test :

- pertes d'entraînement : 0.3542, 0.1667, 0.1165, 0.0887, puis 0.0699 ;
- accuracy : 0.9761 ;
- précision macro : 0.9760 ;
- rappel macro : 0.9758 ;
- F1 macro : 0.9759 ;
- 239 images mal classées ; huit d'entre elles sont affichées ;
- le modèle rechargé depuis `mnist_mlp_state_dict.pt` redonne les mêmes prédictions.

Dans le notebook, ce travail est titré « Exercice 2 : Classifier les chiffres MNIST ».

### Exercice 4 — CNN MNIST

Les images sont normalisées avec la moyenne 0.1307 et l'écart-type 0.3081. Le premier lot a la forme `[64, 1, 28, 28]`.

L'architecture suivie est celle du tableau du notebook :

- convolution 1 → 16 filtres, noyau 3, padding 1, ReLU, max-pooling 2 ;
- convolution 16 → 32 filtres, noyau 3, padding 1, ReLU, max-pooling 2 ;
- aplatissement, puis `Linear(1568, 128)`, ReLU, `Linear(128, 10)`.

La sortie vérifiée sur le premier lot est `[64, 10]`. L'entraînement dure 3 époques avec `CrossEntropyLoss` et Adam au taux 0.001.

Résultats :

- pertes : 0.1680, 0.0482, puis 0.0348 ;
- accuracy sur le test : 98.95 % ;
- dix images mal classées sont affichées avec leur vraie classe et la prédiction ;
- les paramètres du CNN sont enregistrés dans `mnist_cnn.pth`.

La question supplémentaire demande un Dropout avant la dernière couche linéaire. La probabilité n'est pas précisée ; la valeur 0.5 reprend l'exemple de Dropout du cours. Ce second réseau, entraîné lui aussi pendant 3 époques, obtient une perte finale de 0.0768 et une accuracy de 98.76 %. Le fichier `mnist_cnn.pth` correspond au CNN du schéma, sans Dropout.

## Architectures et hyperparamètres

| Modèle | Architecture | Perte | Optimiseur | Entraînement |
| --- | --- | --- | --- | --- |
| Régression du cours | `Linear(1, 1)` | MSE | Adam, lr = 0.05 | 300 époques |
| Classification du cours | `Linear(2, 16)`, ReLU, `Linear(16, 16)`, ReLU, `Linear(16, 2)` | Cross-entropy | Adam, lr = 0.01 | 250 époques |
| Salaire | `Linear(2, 1)` | MSE | SGD, lr = 0.05 | 500 époques |
| Iris | `Linear(4, 16)`, ReLU, `Linear(16, 3)` | Cross-entropy | Adam, lr = 0.01 | 200 époques |
| MLP MNIST | `Flatten`, `Linear(784, 128)`, ReLU, `Linear(128, 10)` | Cross-entropy | Adam, lr = 0.001 | 5 époques, lots de 64 |
| CNN MNIST | Conv 1→16, pool, Conv 16→32, pool, `Linear(1568, 128)`, ReLU, `Linear(128, 10)` | Cross-entropy | Adam, lr = 0.001 | 3 époques, lots de 64 |

## Difficultés rencontrées et solutions

- Deux environnements virtuels étaient présents. `.venv` utilise Python 3.12.10 et PyTorch 2.15.0. `.venv-1`, créé avec Python 3.14, n'était référencé par aucun noyau. Il a été supprimé.
- La cellule des fonctions d'activation a été interrompue pendant `plt.figure()`. Le code était correct : Matplotlib chargeait son backend et son cache de polices. La cellule a été réexécutée avec le backend inline du notebook, et les trois courbes s'affichent.
- `torchviz` a produit `graphe_operations.png`. La commande `dot` de Graphviz est disponible sur la machine.
- Le téléchargement de MNIST échouait avec `SSL: CERTIFICATE_VERIFY_FAILED`. L'installation de `certifi` et l'utilisation de son fichier de certificats ont permis de télécharger les données depuis le miroir prévu par torchvision.
- `torcheval` affiche un avertissement indiquant que `torch.jit.script` est déprécié. Le R² et les métriques binaires ont tout de même été calculés.

## Conclusion

Le notebook couvre le fonctionnement d'un modèle PyTorch, de la création des tenseurs jusqu'à la sauvegarde des poids. Les exemples de la partie 1 retrouvent bien les relations attendues : le poids de la régression est proche de 3, le biais est proche de 2, et les deux classes de points sont séparées.

Sur les exercices, la régression de salaire reste proche de la règle de génération, Iris est parfaitement classé sur le test avec le réseau à couche cachée, le MLP MNIST atteint 97.61 % et le CNN MNIST atteint 98.95 % après les époques demandées.
