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
| :--- | :--- |
| **Python 3.9+** | Langage principal |
| **Pandas** | Manipulation et analyse des données |
| **NumPy** | Calcul numérique |
| **Scikit-Learn** | Machine Learning |
| **Random Forest** | Modèle de classification |
| **Streamlit** | Interface web interactive |
| **Joblib / Pickle** | Sauvegarde et chargement du modèle |

---

## 🤖 Modèle Utilisé

Le modèle principal utilisé dans le projet est un **Random Forest Classifier**.

Le modèle entraîné est sauvegardé dans le fichier :

```text
rf_balanced_model.pkl
```

Le modèle est configuré afin de mieux gérer le déséquilibre potentiel entre les différentes classes de la variable cible (`class_weight='balanced'`).

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

Vérifiez votre version de Python :

```bash
python --version
```

---

### 2. Cloner le projet

```bash
git clone https://github.com/walidlabied09/social-media-impact-prediction.git
cd social-media-impact-prediction
```

---

### 3. Créer un environnement virtuel

**Sous Windows :**

```bash
python -m venv env
.\env\Scripts\activate
```

**Sous Linux / macOS :**

```bash
python3 -m venv env
source env/bin/activate
```

---

### 4. Installer les dépendances

Installez les bibliothèques nécessaires :

```bash
pip install streamlit scikit-learn pandas numpy joblib
```

Optionnel — mise à jour de pip :

```bash
python -m pip install --upgrade pip
```

---

### 5. Lancer l'application

```bash
streamlit run app.py
```

L'application est accessible à l'adresse :

```text
http://localhost:8501
```

---

## 🧠 Modélisation Machine Learning

### Random Forest Classifier

Le projet utilise un **Random Forest Classifier** pour prédire l'intention d'achat des utilisateurs à partir de différentes caractéristiques liées à leur comportement sur les réseaux sociaux.

Cet algorithme d'apprentissage supervisé, basé sur un ensemble d'arbres de décision, permet notamment :

- De gérer des relations non linéaires.
- De travailler avec plusieurs variables explicatives.
- De réduire le risque de surapprentissage par rapport à un arbre unique.
- D'obtenir une mesure interprétable de l'importance des variables.

### ⚖️ Gestion du Déséquilibre des Classes

Le modèle intègre une stratégie de pondération afin d'éviter qu'une classe majoritaire n'écrase les prédictions et d'assurer une bonne détection sur l'ensemble des catégories.

---

## 📊 Évaluation du Modèle

Les performances sont suivies via les métriques standards de classification :

- **Precision :** Proportion de prédictions positives qui sont réellement positives.
- **Recall :** Capacité du modèle à identifier l'ensemble des observations positives.
- **F1-Score :** Moyenne harmonique de la précision et du rappel pour une vision équilibrée de la performance.

---

## 🔎 Feature Importance

Le modèle permet d'extraire les variables prédictives les plus déterminantes dans l'acte d'achat :

- Temps passé quotidiennement sur les plateformes.
- Plateformes utilisées (Instagram, TikTok, YouTube, etc.).
- Type de contenu consulté.
- Degré de réceptivité aux recommandations d'influenceurs.
- Fréquence globale d'utilisation.

---

## 🌐 Application Streamlit

L'interface web permet à un utilisateur de saisir interactivement ses paramètres pour obtenir une prédiction immédiate :

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

## 🚀 Flux d'Utilisation

```text
Utilisateur
    │
    ▼
Variables liées aux réseaux sociaux
    │
    ▼
Modèle Random Forest
    │
    ▼
Prédiction du comportement d'achat
```

---

## 🔐 Fichiers Ignorés (.gitignore)

Le fichier `.gitignore` doit contenir :

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

- Tester d'autres familles d'algorithmes (XGBoost, LightGBM, régression logistique pénalisée).
- Automatiser la recherche d'hyperparamètres (GridSearchCV, Optuna).
- Intégrer une validation croisée k-fold stratifiée.
- Afficher une matrice de confusion et les courbes ROC interactives dans Streamlit.
- Intégrer un graphique interactif de Feature Importance directement dans l'interface.
- Déployer l'application sur Streamlit Community Cloud.

---

## 🎯 Cycle du Projet

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

---

## 👤 Auteur

**Walid Labied**

*Data & Software Engineering*

GitHub : [walidlabied09](https://github.com/walidlabied09)

---

## 📄 Licence

Ce projet est réalisé dans un cadre académique et de démonstration technique en Data Science et Machine Learning.
