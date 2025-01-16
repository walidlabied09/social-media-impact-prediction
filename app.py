import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie

# Tests statistiques
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, pearsonr

# Mesures ML
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Préparation ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Feature Selection
from sklearn.feature_selection import SequentialFeatureSelector as SFS

# 3 Algorithmes
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Sauvegarde / Chargement de modèle
import pickle

# Gestion du déséquilibre pour Random Forest
from imblearn.over_sampling import RandomOverSampler

# -------------------------------------------------------------------
#                FONCTIONS UTILES
# -------------------------------------------------------------------
def load_lottieurl(url: str):
    """
    Charge un fichier d'animation Lottie depuis une URL.
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_data
def load_data():
    """
    Lit le fichier Excel et renvoie un DataFrame brut,
    en convertissant toute colonne datetime en string
    pour éviter l'erreur "Cannot cast DatetimeArray to dtype float64".
    """
    df = pd.read_excel('Impact of Social Media on Buying Behavior.(1-178).xlsx')
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df

def preprocess_data(df):
    """
    Transforme 'Gender', 'Income', 'Platform' en dummies, 
    scale 'Age' et 'Hours', supprime 'Age'/'Hours' bruts.
    """
    df = df.fillna(0)

    # Encodage
    if 'Gender' in df.columns and \
       'What is your approximate monthly income?' in df.columns and \
       'Which social media platform do you use the most?' in df.columns:
        df = pd.get_dummies(
            df,
            columns=[
                'Gender',
                'What is your approximate monthly income?',
                'Which social media platform do you use the most?'
            ],
            drop_first=False
        )

    # Mise à l'échelle
    if 'Age' in df.columns and 'How many hours do you spend on social media per day?' in df.columns:
        scaler = StandardScaler()
        df[['Age_scaled','Hours_scaled']] = scaler.fit_transform(
            df[['Age','How many hours do you spend on social media per day?']]
        )
        df.drop(['Age','How many hours do you spend on social media per day?'], axis=1, inplace=True)

    return df

def prepare_X_y(df):
    """
    y = 'Have you ever purchased...?' (Yes=1 / No=0)
    X = colonnes dummies + Age_scaled + Hours_scaled
    """
    y = df['Have you ever purchased a product or service after seeing an advertisement or recommendation on social media?'] \
        .map({'Yes':1,'No':0})

    needed_cols = [
        'Age_scaled', 
        'Hours_scaled',
        'Gender_Man', 
        'Gender_Woman',
        'What is your approximate monthly income?_Between 5000 and 10000 MAD',
        'What is your approximate monthly income?_Between 10000 and 20000 MAD',
        'What is your approximate monthly income?_Less than 5000 MAD',
        'Which social media platform do you use the most?_Facebook', 
        'Which social media platform do you use the most?_Instagram',
        'Which social media platform do you use the most?_Twitter', 
        'Which social media platform do you use the most?_TikTok'
    ]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[needed_cols].copy()
    return X, y

# -------------------------------------------------------------------
#    NOUVELLES FONCTIONS POUR MODÈLE RANDOM FOREST BALANCÉ
# -------------------------------------------------------------------
def train_balanced_model_rf(X, y):
    """
    Entraîne un RandomForestClassifier avec oversampling
    et sauvegarde dans 'rf_balanced_model.pkl'.
    """
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Oversampling
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    # Modèle Random Forest (paramètres par défaut, n_estimators=100)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)

    # Sauvegarde
    with open('rf_balanced_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

def load_balanced_model_rf():
    """
    Charge rf_balanced_model.pkl si existant.
    """
    try:
        with open('rf_balanced_model.pkl','rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning("Le modèle RF Balanced n'a pas encore été entraîné.")
        return None

def predict_purchase_rf(model, age, gender, income, platform, hours):
    """
    Construit un mini-DataFrame (1 ligne),
    encode + scale localement,
    puis model.predict() => 0 ou 1
    """
    # Construction du mini-DataFrame
    df_input = pd.DataFrame(
        [[age, gender, income, platform, hours]],
        columns=[
            'Age',
            'Gender',
            'What is your approximate monthly income?',
            'Which social media platform do you use the most?',
            'Hours'
        ]
    )
    # Get dummies
    df_input = pd.get_dummies(
        df_input,
        columns=[
            'Gender',
            'What is your approximate monthly income?',
            'Which social media platform do you use the most?'
        ],
        drop_first=False
    )
    # Mise à l'échelle locale
    sc = StandardScaler()
    if 'Age' in df_input.columns and 'Hours' in df_input.columns:
        df_input[['Age_scaled','Hours_scaled']] = sc.fit_transform(
            df_input[['Age','Hours']]
        )
        df_input.drop(['Age','Hours'], axis=1, inplace=True)

    needed_cols = [
        'Age_scaled',
        'Hours_scaled',
        'Gender_Man', 
        'Gender_Woman',
        'What is your approximate monthly income?_Between 5000 and 10000 MAD',
        'What is your approximate monthly income?_Between 10000 and 20000 MAD',
        'What is your approximate monthly income?_Less than 5000 MAD',
        'Which social media platform do you use the most?_Facebook', 
        'Which social media platform do you use the most?_Instagram',
        'Which social media platform do you use the most?_Twitter', 
        'Which social media platform do you use the most?_TikTok'
    ]
    for col in needed_cols:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[needed_cols]

    pred = model.predict(df_input)
    return pred[0]

# -------------------------------------------------------------------
#         CONFIGURATION DE LA PAGE & STYLE GÉNÉRAL
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Social Media Impact - Modern",
    page_icon=":sparkles:",
    layout="wide"
)

# Palette de couleurs commune
COLOR_SEQUENCE = px.colors.qualitative.Prism

# CSS personnalisé
st.markdown(
    """
    <style>
    /* Masquer le menu et le footer Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #FAFAFA;
        color: #4a4a4a;
    }

    .big-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #333333;
        text-align: center;
        margin-top: 10px;
    }

    .small-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #4a4a4a;
        margin: 25px 0 10px;
    }
    .highlight {
        color: #FF7F50;
        font-weight: 700;
    }
    .stContainer {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    button[kind="primary"] {
        font-size:1rem !important;
        padding:0.6rem 1.2rem !important;
    }
    .stRadio > label, .stCheckbox > label {
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    # Chargement d'une animation Lottie (optionnel)
    lottie_animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_5ngs2ksb.json")

    # Container principal
    with st.container():
        # Animation Lottie en haut
        col_animation, col_title = st.columns([1,3])
        with col_animation:
            if lottie_animation:
                from streamlit_lottie import st_lottie
                st_lottie(lottie_animation, key="lottie_anim", height=180)
        with col_title:
            st.markdown("<h1 class='big-title'>Impact of Social Media on Buying Behavior</h1>", unsafe_allow_html=True)

        data = load_data()

    # -------------------  SIDEBAR  -------------------
    st.sidebar.title("Navigation")
    user_page = st.sidebar.checkbox("Page Utilisateur (RF Balanced)")
    page = st.sidebar.radio(
        "Aller à la section :",
        ["Graphiques", "Tests d'Hypothèses", "Machine Learning"]
    )

    # ------------------- PAGE UTILISATEUR -------------------
    if user_page:
        with st.container():
            st.markdown("<h2 class='small-title'>Page Utilisateur : <span class='highlight'>Prédiction (RF Balanced)</span></h2>", unsafe_allow_html=True)
            col_train, col_predict = st.columns([1,2])

            # Entraînement du modèle RF Balancé
            with col_train:
                if st.button("Entraîner le Modèle RF Balanced"):
                    data_prep = preprocess_data(data)
                    X, y = prepare_X_y(data_prep)
                    model_rf = train_balanced_model_rf(X, y)
                    st.success("Modèle RF Balanced entraîné et sauvegardé sous 'rf_balanced_model.pkl'.")

            # Formulaire de prédiction
            with col_predict:
                st.markdown("*Formulaire de Prédiction*")
                age_val = st.number_input("Âge", min_value=18, max_value=100, value=25)
                gender_val = st.selectbox("Genre", ["Man","Woman"])
                income_val = st.selectbox("Revenu Mensuel", [
                    "Less than 5000 MAD",
                    "Between 5000 and 10000 MAD",
                    "Between 10000 and 20000 MAD"
                ])
                platform_val = st.selectbox("Plateforme principale", [
                    "Facebook",
                    "Instagram",
                    "Twitter",
                    "TikTok"
                ])
                hours_val = st.slider("Heures/jour sur RS", 0, 24, 2)

                # Bouton de prédiction RF
                if st.button("Prédire (RF Balanced)"):
                    model_rf_bal = load_balanced_model_rf()
                    if model_rf_bal:
                        pred = predict_purchase_rf(
                            model_rf_bal,
                            age_val,
                            gender_val,
                            income_val,
                            platform_val,
                            hours_val
                        )
                        if pred == 1:
                            st.success("Vous êtes susceptible d'acheter après pub/reco (classe 1).")
                        else:
                            st.info("Vous êtes moins susceptible d'acheter après pub/reco (classe 0).")

    # ------------------- SECTION GRAPHIQUES -------------------
    if page == "Graphiques":
        st.markdown("<h2 class='small-title'>🎨 Graphiques</h2>", unsafe_allow_html=True)
        
        # Définir la palette de couleurs au début
        COLOR_SEQUENCE = px.colors.sequential.Viridis

        col_g1, col_g2 = st.columns([1, 3])

        with col_g1:
            graph_choice = st.radio(
                "Sélectionnez un graphique :", (
                    "Distribution des Âges",
                    "Répartition des Genres",
                    "Statut Actuel",
                    "Répartition des Revenus Mensuels",
                    "Plateforme de Réseaux Sociaux",
                    "Heures Passées sur les Réseaux Sociaux",
                    "Achats Après Publicité",
                    "Suivi des Influenceurs",
                    "Facteurs d'Achat",
                    "Nombre d'Achats (6 derniers mois)",
                    "Confiance dans les Avis (1 à 10)"
                )
            )

        with col_g2:
            st.markdown("*Visualisation :*")

            if graph_choice == "Distribution des Âges":
                if 'Age' in data.columns:
                    fig = px.histogram(
                        data_frame=data,
                        x='Age',
                        nbins=20,
                        title="Distribution de l'Âge",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        template="simple_white"
                    )
                    fig.update_layout(
                        bargap=0.2,
                        title_font=dict(size=20, family='Arial', color='#4a4a4a'),
                        xaxis=dict(title="Âge", titlefont=dict(size=16)),
                        yaxis=dict(title="Fréquence", titlefont=dict(size=16)),
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    fig.update_traces(marker=dict(line=dict(width=0.5, color='black')))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Colonne 'Age' introuvable.")

            elif graph_choice == "Répartition des Genres":
                if 'Gender' in data.columns:
                    g_counts = data['Gender'].value_counts()
                    fig = px.pie(
                        names=g_counts.index,
                        values=g_counts.values,
                        title="Répartition des Genres",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        template="simple_white"
                    )
                    fig.update_traces(
                        textinfo='percent+label',
                        marker=dict(line=dict(color='#FFFFFF', width=2))
                    )
                    fig.update_layout(
                        title=dict(font=dict(size=20, color='#4a4a4a')),
                        legend=dict(title="Genres", font=dict(size=14)),
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Colonne 'Gender' introuvable.")

            elif graph_choice == "Statut Actuel":
                col_statut = 'What is your current status?'
                if col_statut in data.columns:
                    s_counts = data[col_statut].value_counts()
                    fig = px.pie(
                        names=s_counts.index,
                        values=s_counts.values,
                        title="Statut Actuel",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        template="simple_white"
                    )
                    fig.update_layout(
                        title_font=dict(size=20, family='Arial', color='#4a4a4a'),
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Colonne 'What is your current status?' introuvable.")

            elif graph_choice == "Répartition des Revenus Mensuels":
                col_inc = 'What is your approximate monthly income?'
                if col_inc in data.columns:
                    inc_counts = data[col_inc].value_counts().sort_index()
                    fig = px.bar(
                        x=inc_counts.index,
                        y=inc_counts.values,
                        title="Répartition des Revenus Mensuels",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        template="simple_white"
                    )
                    fig.update_layout(
                        title_font=dict(size=20, family='Arial', color='#4a4a4a'),
                        xaxis=dict(title="Catégories de Revenus", titlefont=dict(size=16)),
                        yaxis=dict(title="Fréquence", titlefont=dict(size=16)),
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    fig.update_traces(marker=dict(line=dict(width=0.5, color='black')))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Colonne 'What is your approximate monthly income?' introuvable.")

            elif graph_choice == "Plateforme de Réseaux Sociaux":
                col_plat = 'Which social media platform do you use the most?'
                if col_plat in data.columns:
                    p_counts = data[col_plat].value_counts()
                    fig = px.bar(
                        x=p_counts.index,
                        y=p_counts.values,
                        title="Plateforme la plus utilisée",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        template="simple_white"
                    )
                    fig.update_layout(
                        title_font=dict(size=20, family='Arial', color='#4a4a4a'),
                        xaxis=dict(title="Plateformes", titlefont=dict(size=16)),
                        yaxis=dict(title="Fréquence", titlefont=dict(size=16)),
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    fig.update_traces(marker=dict(line=dict(width=0.5, color='black')))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Colonne 'Which social media platform do you use the most?' introuvable.")

            elif graph_choice == "Heures Passées sur les Réseaux Sociaux":
                col_hours = 'How many hours do you spend on social media per day?'
                if col_hours in data.columns:
                    fig = px.histogram(
                        data_frame=data,
                        x=col_hours,
                        nbins=10,
                        title="Heures Passées / Jour",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        template="simple_white"
                    )
                    fig.update_layout(
                        title_font=dict(size=20, family='Arial', color='#4a4a4a'),
                        xaxis=dict(title="Heures", titlefont=dict(size=16)),
                        yaxis=dict(title="Fréquence", titlefont=dict(size=16)),
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    fig.update_traces(marker=dict(line=dict(width=0.5, color='black')))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Colonne '{col_hours}' introuvable.")

            elif graph_choice == "Achats Après Publicité":
                col_apub = 'Have you ever purchased a product or service after seeing an advertisement or recommendation on social media?'
                if col_apub in data.columns:
                    a_counts = data[col_apub].value_counts()
                    fig = px.pie(
                        names=a_counts.index,
                        values=a_counts.values,
                        title="Achat Après Publicité/Reco ?",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        template="simple_white"
                    )
                    fig.update_traces(
                        textinfo='percent+label',
                        marker=dict(line=dict(color='#FFFFFF', width=2))
                    )
                    fig.update_layout(
                        title_font=dict(size=20, family='Arial', color='#4a4a4a'),
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Colonne '{col_apub}' introuvable.")

            elif graph_choice == "Suivi des Influenceurs":
                col_influ = 'Do you follow influencers or pages dedicated to specific products/services on social media?'
                if col_influ in data.columns:
                    i_counts = data[col_influ].value_counts()
                    fig = px.pie(
                        names=i_counts.index,
                        values=i_counts.values,
                        title="Suivi Influenceurs / Pages Produits ?",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        template="simple_white"
                    )
                    fig.update_traces(
                        textinfo='percent+label',
                        marker=dict(line=dict(color='#FFFFFF', width=2))
                    )
                    fig.update_layout(
                        title_font=dict(size=20, family='Arial', color='#4a4a4a'),
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Colonne '{col_influ}' introuvable.")

            elif graph_choice == "Facteurs d'Achat":
                col_factor = 'Which factor influence you the most when buying a product recommended on social media?'
                if col_factor in data.columns:
                    fac_counts = data[col_factor].value_counts()
                    fig = px.bar(
                        x=fac_counts.index,
                        y=fac_counts.values,
                        title="Facteurs influençant l'achat",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        template="simple_white"
                    )
                    fig.update_layout(
                        xaxis=dict(title="Facteurs", titlefont=dict(size=16)),
                        yaxis=dict(title="Fréquence", titlefont=dict(size=16)),
                        title_font=dict(size=20, family='Arial', color='#4a4a4a'),
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Colonne '{col_factor}' introuvable.")

            elif graph_choice == "Nombre d'Achats (6 derniers mois)":
                col_times = 'How many times have you purchased a product recommended on social media in the last 6 months?'
                if col_times in data.columns:
                    fig = px.histogram(
                        data_frame=data,
                        x=col_times,
                        nbins=10,
                        title="Achats (6 derniers mois)",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        template="simple_white"
                    )
                    fig.update_layout(
                        title_font=dict(size=20, family='Arial', color='#4a4a4a'),
                        xaxis=dict(title="Nombre d'Achats", titlefont=dict(size=16)),
                        yaxis=dict(title="Fréquence", titlefont=dict(size=16)),
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Colonne '{col_times}' introuvable.")

            elif graph_choice == "Confiance dans les Avis (1 à 10)":
                col_trust = 'On a scale of 1 to 10, how much do you trust consumer reviews on social media before purchasing a product?'
                if col_trust in data.columns:
                    fig = px.histogram(
                        data_frame=data,
                        x=col_trust,
                        nbins=10,
                        title="Confiance (1=faible, 10=fort)",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        template="simple_white"
                    )
                    fig.update_layout(
                        title_font=dict(size=20, family='Arial', color='#4a4a4a'),
                        xaxis=dict(title="Confiance", titlefont=dict(size=16)),
                        yaxis=dict(title="Fréquence", titlefont=dict(size=16)),
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Colonne '{col_trust}' introuvable.")

    # ------------------- SECTION TESTS D'HYPOTHESES -------------------
    elif page == "Tests d'Hypothèses":
        st.markdown("<h2 class='small-title'>🔠 Tests d'Hypothèses</h2>", unsafe_allow_html=True)
        st.markdown("Effectuez des tests statistiques pour valider ou rejeter vos hypothèses.")

        # Test du Chi-2
        st.subheader("1. Test du Chi-2")
        st.markdown("*H0 :* Suivre des influenceurs n'a aucun effet significatif sur les probabilités d'achat après avoir vu une publicité.")
        st.markdown("*H1 :* Suivre des influenceurs a un effet significatif sur les probabilités d'achat.")

        df_clean = data[[
            'Do you follow influencers or pages dedicated to specific products/services on social media?',
            'Have you ever purchased a product or service after seeing an advertisement or recommendation on social media?'
        ]].dropna()

        contingency_table = pd.crosstab(
            df_clean['Do you follow influencers or pages dedicated to specific products/services on social media?'],
            df_clean['Have you ever purchased a product or service after seeing an advertisement or recommendation on social media?']
        )
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        st.write("*Table de contingence :*")
        st.dataframe(contingency_table)
        st.write(f"*Chi-Carré Statistique :* {chi2}")
        st.write(f"*p-value :* {p}")
        st.write(f"*Degrés de Liberté :* {dof}")

        if p < 0.05:
            st.success("Rejet de H0. Suivre des influenceurs augmente significativement les probabilités d'achat.")
        else:
            st.info("Ne pas rejeter H0. Aucune influence significative détectée.")

        # Test T et ANOVA
        st.subheader("2. Test T et ANOVA")
        st.markdown("*H0 :* Le temps passé sur les réseaux sociaux n'influence pas significativement le nombre d'achats.")
        st.markdown("*H1 :* Le temps passé sur les réseaux sociaux influence significativement le nombre d'achats.")

        df_clean = data[[
            'How many hours do you spend on social media per day?',
            'How many times have you purchased a product recommended on social media in the last 6 months?'
        ]].dropna()

        group_1 = df_clean[df_clean['How many hours do you spend on social media per day?'] <= 2]['How many times have you purchased a product recommended on social media in the last 6 months?']
        group_2 = df_clean[(df_clean['How many hours do you spend on social media per day?'] > 2) & (df_clean['How many hours do you spend on social media per day?'] <= 5)]['How many times have you purchased a product recommended on social media in the last 6 months?']
        group_3 = df_clean[df_clean['How many hours do you spend on social media per day?'] > 5]['How many times have you purchased a product recommended on social media in the last 6 months?']

        t_stat, p_value_ttest = ttest_ind(group_1, group_2)
        f_stat, p_value_anova = f_oneway(group_1, group_2, group_3)

        st.write("*Résultats du Test T :*")
        st.write(f"Statistique T : {t_stat}")
        st.write(f"p-value : {p_value_ttest}")
        if p_value_ttest < 0.05:
            st.success("Rejet de H0. Le temps passé sur les réseaux sociaux influence significativement les achats.")
        else:
            st.info("Ne pas rejeter H0. Aucune influence significative détectée.")

        st.write("*Résultats de l'ANOVA :*")
        st.write(f"Statistique F : {f_stat}")
        st.write(f"p-value : {p_value_anova}")
        if p_value_anova < 0.05:
            st.success("Rejet de H0. Les groupes ont des moyennes d'achats significativement différentes.")
        else:
            st.info("Ne pas rejeter H0. Pas de différences significatives entre les groupes.")

        # Corrélation Pearson
        st.subheader("3. Test de Corrélation (Pearson)")
        st.markdown("*H0 :* La confiance dans les avis des consommateurs n'est pas corrélée de manière significative avec les achats.")
        st.markdown("*H1 :* La confiance dans les avis des consommateurs est corrélée de manière significative avec les achats.")

        df_clean = data[[
            'On a scale of 1 to 10, how much do you trust consumer reviews on social media before purchasing a product?',
            'How many times have you purchased a product recommended on social media in the last 6 months?'
        ]].dropna()

        corr, p_value = pearsonr(
            df_clean['On a scale of 1 to 10, how much do you trust consumer reviews on social media before purchasing a product?'],
            df_clean['How many times have you purchased a product recommended on social media in the last 6 months?']
        )

        st.write(f"*Corrélation de Pearson :* {corr}")
        st.write(f"*p-value :* {p_value}")

        if p_value < 0.05:
            st.success("Rejet de H0. La confiance dans les avis des consommateurs est corrélée significativement avec les achats.")
        else:
            st.info("Ne pas rejeter H0. Aucune corrélation significative.")

    # ------------------- SECTION MACHINE LEARNING -------------------
    elif page == "Machine Learning":
        st.markdown("<h2 class='small-title'>🧬 Machine Learning</h2>", unsafe_allow_html=True)
        st.subheader("Prédictions des Modèles")

        # Charger les données
        df = load_data()
        # Preprocessing
        y1 = df['Have you ever purchased a product or service after seeing an advertisement or recommendation on social media?'].map({'Yes': 1, 'No': 0})
        X = df[['Age', 'Gender', 'What is your approximate monthly income?', 'Which social media platform do you use the most?', 'How many hours do you spend on social media per day?']].copy()
        X = X.fillna(0)
        X = pd.get_dummies(X, columns=['Gender', 'What is your approximate monthly income?', 'Which social media platform do you use the most?'])
        scaler = StandardScaler()
        X[['Age_scaled', 'Hours_scaled']] = scaler.fit_transform(X[['Age', 'How many hours do you spend on social media per day?']])
        X.drop(['Age', 'How many hours do you spend on social media per day?'], axis=1, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.3, random_state=42, stratify=y1)

        # Sélection de features spécifiques (arbitraires)
        features = {
            "Logistic Regression": [
                'Gender_Man',
                'Gender_Woman',
                'What is your approximate monthly income?_Between 10000 and 20000 MAD',
                'What is your approximate monthly income?_Between 5000 and 10000 MAD',
                'What is your approximate monthly income?_Less than 5000 MAD'
            ],
            "Random Forest": [
                'Hours_scaled',
                'Age_scaled',
                'Which social media platform do you use the most?_Instagram',
                'Which social media platform do you use the most?_Facebook',
                'Gender_Man'
            ],
            "Support Vector Machine": [
                'Hours_scaled',
                'Age_scaled'
            ]
        }

        # Initialisation des modèles
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Support Vector Machine": SVC(probability=True, random_state=42)
        }

        # Entraîner et évaluer chaque modèle
        for name, model in models.items():
            st.write(f"Entraînement de *{name}*...")
            X_train_model = X_train[features[name]]
            X_test_model  = X_test[features[name]]
            model.fit(X_train_model, y_train)

            # Prédictions
            y_pred = model.predict(X_test_model)
            y_prob = model.predict_proba(X_test_model)[:, 1] if hasattr(model, "predict_proba") else None

            # Affichage
            st.write(f"--- *{name}* ---")
            st.write(f"Précision : {accuracy_score(y_test, y_pred):.4f}")
            st.write(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
            if y_prob is not None:
                st.write(f"ROC-AUC Score : {roc_auc_score(y_test, y_prob):.4f}")

            # Rapport de classification
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            st.write("*Classification Report :*")
            st.dataframe(pd.DataFrame(classification_rep).transpose())

        # Matrice de confusion pour Logistic Regression
        st.subheader("Matrice de Confusion (Logistic Regression)")
        model_lr = LogisticRegression()
        model_lr.fit(X_train[features["Logistic Regression"]], y_train)
        y_pred_lr = model_lr.predict(X_test[features["Logistic Regression"]])
        cm = confusion_matrix(y_test, y_pred_lr)
        st.table(pd.DataFrame(cm, columns=["Prédit: 0", "Prédit: 1"], index=["Réel: 0", "Réel: 1"]))

        # Sélection de caractéristiques (SFS)
        st.subheader("Sélection de caractéristiques (Forward Selection)")
        sfs = SFS(
            estimator=LogisticRegression(),  
            n_features_to_select=5,        
            direction='forward',
            scoring='accuracy',
            cv=5
        )
        sfs = sfs.fit(X, y1)
        selected_indices = sfs.get_support()
        selected_features = X.columns[selected_indices]
        st.write("Caractéristiques sélectionnées par sélection avant :", list(selected_features))

        # Importance des caractéristiques (Random Forest)
        st.subheader("Importance des caractéristiques (Random Forest)")
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X, y1)
        importances = rf.feature_importances_
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        st.write("Top des caractéristiques par importance (Random Forest) :")
        st.write(importance_df.head(5))

        # Vérifiez que le DataFrame importance_df contient les colonnes nécessaires
        if 'Feature' in importance_df.columns and 'Importance' in importance_df.columns:
            COLOR_SEQUENCE = px.colors.sequential.Viridis
            fig = px.bar(
                importance_df.head(10).sort_values("Importance", ascending=True),
                x="Importance",
                y="Feature",
                orientation='h',
                title="Importance des Caractéristiques (Random Forest)",
                color_discrete_sequence=COLOR_SEQUENCE
            )
            fig.update_layout(
                title_font=dict(size=20, family='Arial', color='#4a4a4a'),
                xaxis=dict(title="Importance", titlefont=dict(size=16)),
                yaxis=dict(title="Caractéristiques", titlefont=dict(size=16)),
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Les colonnes 'Feature' ou 'Importance' sont introuvables dans le DataFrame.")

# -------------------------------------------------------------------
#               LANCEMENT DE L'APPLICATION
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
