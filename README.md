# 📊 Impact of Social Media on Buying Behavior — ML Prediction

Application d'analyse prédictive et de Machine Learning permettant d'évaluer et de prédire l'impact de l'exposition aux réseaux sociaux sur les décisions d'achat des consommateurs.

---

## 🎯 Objectifs du Projet

- **Analyse Exploratoire des Données (EDA) :** Identifier les facteurs clés des réseaux sociaux (temps passé, plateformes utilisées, influenceurs, types de contenu) influençant l'acte d'achat.
- **Modélisation Prédictive :** Entraîner un classifieur supervisé robuste capable de prédire l'intention d'achat tout en gérant le déséquilibre des classes.
- **Déploiement Interactif :** Mettre à disposition une interface web interactive permettant de simuler des prédictions en temps réel selon le profil utilisateur.

---

## 🛠️ Stack Technique

| Technologie | Utilisation |
|---|---|
| **Python 3.9+** | Langage principal |
| **Pandas** | Manipulation et analyse des données |
| **NumPy** | Calcul numérique |
| **Scikit-Learn** | Machine Learning |
| **Random Forest** | Modèle de classification |
| **Streamlit** | Interface web interactive |
| **Joblib / Pickle** | Sauvegarde et chargement du modèle |

### 🤖 Modèle utilisé

Le modèle principal utilisé dans le projet est un :

**Random Forest Classifier**

Le modèle entraîné est sauvegardé dans le fichier :

```text
rf_balanced_model.pkl
```

Le modèle est configuré afin de mieux gérer le déséquilibre potentiel entre les différentes classes de la variable cible.

---

## 📁 Structure du Répertoire

```text
social-media-impact-prediction/
│
├── Impact of Social Media on Buying Behavior.(1-1...
│   └── Dataset utilisé pour l'étude
│
├── app.py
│   └── Application Streamlit et interface de prédiction
│
├── rf_balanced_model.pkl
│   └── Modèle Random Forest entraîné
│
├── .gitignore
│   └── Fichiers et dossiers exclus du dépôt Git
│
└── README.md
    └── Documentation du projet
```

---

## ⚙️ Installation & Démarrage

### 1. Prérequis

Avant de commencer, assurez-vous d'avoir installé :

- Python 3.9 ou supérieur
- Git
- pip

Vous pouvez vérifier votre version de Python avec :

```bash
python --version
```

---

### 2. Cloner le projet

```bash
git clone https://github.com/walidlabied09/social-media-impact-prediction.git
```

Puis accéder au répertoire :

```bash
cd social-media-impact-prediction
```

---

### 3. Créer un environnement virtuel

Sous Windows :

```bash
python -m venv env
```

Activer l'environnement virtuel :

```bash
.\env\Scripts\activate
```

Sous Linux / macOS :

```bash
python3 -m venv env
```

Puis :

```bash
source env/bin/activate
```

---

### 4. Installer les dépendances

Installer les principales bibliothèques nécessaires :

```bash
pip install streamlit scikit-learn pandas numpy joblib
```

Vous pouvez également mettre pip à jour :

```bash
python -m pip install --upgrade pip
```

---

### 5. Lancer l'application

Une fois les dépendances installées, exécuter :

```bash
streamlit run app.py
```

L'application sera normalement accessible à l'adresse :

```text
http://localhost:8501
```

---

## 🧠 Modélisation Machine Learning

### Random Forest Classifier

Le projet utilise un **Random Forest Classifier** pour prédire l'intention d'achat des utilisateurs à partir de différentes caractéristiques liées à leur utilisation des réseaux sociaux.

Le Random Forest est un algorithme d'apprentissage supervisé basé sur un ensemble d'arbres de décision.

Il permet notamment :

- de gérer des relations non linéaires ;
- de travailler avec plusieurs variables explicatives ;
- de réduire le risque de surapprentissage par rapport à un arbre unique ;
- d'obtenir une mesure de l'importance des variables.

---

## ⚖️ Gestion du Déséquilibre des Classes

Le modèle est conçu pour prendre en compte un éventuel déséquilibre entre les classes de la variable cible.

L'objectif est d'éviter qu'une classe majoritaire domine les prédictions du modèle et d'améliorer la capacité du classifieur à identifier correctement les différentes catégories.

Le fichier du modèle entraîné est :

```text
rf_balanced_model.pkl
```

---

## 📊 Évaluation du Modèle

Les performances du modèle peuvent être évaluées à l'aide de plusieurs métriques de classification :

### Precision

Mesure la proportion de prédictions positives qui sont réellement positives.

### Recall

Mesure la capacité du modèle à identifier correctement les observations positives.

### F1-Score

Combine la précision et le rappel afin de fournir une mesure globale de la performance du modèle.

---

## 🔎 Feature Importance

Le Random Forest permet également d'analyser l'importance des différentes variables utilisées pour effectuer les prédictions.

Cette analyse permet notamment d'identifier les facteurs liés aux réseaux sociaux qui contribuent le plus à la prédiction du comportement d'achat.

Par exemple :

- temps passé sur les réseaux sociaux ;
- plateforme utilisée ;
- type de contenu consulté ;
- influence des influenceurs ;
- fréquence d'utilisation ;
- comportement de l'utilisateur.

---

## 🌐 Application Streamlit

L'application développée avec **Streamlit** fournit une interface interactive permettant à l'utilisateur de renseigner différentes caractéristiques de son profil.

Le système utilise ensuite le modèle Random Forest entraîné afin de générer une prédiction.

### Fonctionnement général

```text
Profil utilisateur
        │
        ▼
Saisie des caractéristiques
        │
        ▼
Prétraitement des données
        │
        ▼
Random Forest Classifier
        │
        ▼
Prédiction
        │
        ▼
Résultat affiché dans Streamlit
```

---

## 🚀 Exemple d'utilisation

Après avoir lancé :

```bash
streamlit run app.py
```

l'utilisateur peut accéder à l'interface web et renseigner les informations demandées.

L'application transmet ensuite les données au modèle :

```text
Utilisateur
    ↓
Variables liées aux réseaux sociaux
    ↓
Modèle Random Forest
    ↓
Prédiction du comportement d'achat
```

---

## 📦 Dépendances principales

Les principales bibliothèques utilisées dans ce projet sont :

```text
streamlit
scikit-learn
pandas
numpy
joblib
```

Pour installer toutes les dépendances :

```bash
pip install streamlit scikit-learn pandas numpy joblib
```

---

## 🔐 Fichiers Git à ignorer

Le fichier `.gitignore` peut notamment contenir :

```gitignore
__pycache__/
*.pyc
.env
env/
venv/
.venv/
.ipynb_checkpoints/
```

---

## 📈 Perspectives d'Amélioration

Plusieurs améliorations peuvent être envisagées :

- Tester d'autres algorithmes de classification.
- Optimiser les hyperparamètres du Random Forest.
- Ajouter une validation croisée.
- Comparer plusieurs modèles avec différentes métriques.
- Ajouter une matrice de confusion interactive.
- Visualiser l'importance des variables directement dans Streamlit.
- Ajouter des graphiques d'analyse exploratoire.
- Déployer l'application sur Streamlit Community Cloud.
- Ajouter une pipeline complète de prétraitement des données.
- Améliorer l'interface utilisateur et l'expérience utilisateur.

---

## 🎯 Résultat

Ce projet permet de mettre en pratique différentes étapes d'un projet Data Science :

```text
Collecte des données
        ↓
Nettoyage des données
        ↓
Analyse exploratoire (EDA)
        ↓
Préparation des données
        ↓
Entraînement du modèle
        ↓
Évaluation
        ↓
Sauvegarde du modèle
        ↓
Application Streamlit
        ↓
Prédiction interactive
```

Il constitue ainsi une démonstration complète d'une approche **Machine Learning + Data Analysis + Application Web**.

---

## 👤 Auteur

**Walid Labied**

Data & Software Engineering

GitHub :  
https://github.com/walidlabied09

---

## 📄 Licence

Ce projet est réalisé dans un objectif académique et de démonstration des compétences en **Data Science, Machine Learning et développement d'applications interactives**.
