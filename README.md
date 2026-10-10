# Projet 3 — Anticipez les besoins en consommation de bâtiments de Seattle

> Formation **OpenClassrooms — AI Engineer** · Modélisation supervisée · Data Scientist

Seattle vise la **neutralité carbone en 2050** et doit connaître la consommation d'énergie et les
émissions de son parc **non résidentiel**. Les relevés terrain étant coûteux, la Ville veut
**estimer** ces grandeurs pour les bâtiments non mesurés à partir des seules caractéristiques
structurelles des registres municipaux.

**Deux cibles à prédire :**

| Cible | Description | Unité |
|---|---|---|
| `SiteEnergyUse(kBtu)` | Consommation totale d'énergie | kBtu/an |
| `TotalGHGEmissions` | Émissions totales de gaz à effet de serre | t CO₂e/an |

## Sommaire

- [Contexte et objectif](#contexte-et-objectif)
- [Jeu de données](#jeu-de-données)
- [Démarche](#démarche)
- [Résultats](#résultats)
- [Enseignements clés](#enseignements-clés)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation et exécution](#installation-et-exécution)
- [Limites et pistes](#limites-et-pistes)
- [Ressources](#ressources)

## Contexte et objectif

Mission de data science supervisée : reprendre une analyse préparée par Léa, sous la coordination
de Douglas, pour **réduire le coût des relevés terrain** en construisant un modèle prédictif fiable
sur les bâtiments dont les mesures ne sont pas encore disponibles.

Le périmètre suit le champ `BuildingType` (classification de la Ville) : les **bâtiments non
destinés à l'habitation**. Les cas limites (résidences universitaires, résidences seniors) sont
signalés plutôt qu'arbitrés à la main.

## Jeu de données

- **Source officielle :** [Building Energy Benchmarking Data (Seattle)](https://data.seattle.gov/Built-Environment/Building-Energy-Benchmarking-Data-2015-Present/teqw-tu6e/about_data)
- **Extrait utilisé :** campagne **2016** — `2016_Building_Energy_Benchmarking.csv`
- **Téléchargement :** automatique via `wget` à la première cellule du notebook
- **Périmètre après nettoyage :** **1 546 bâtiments** (1 236 entraînement / 310 test), 15 variables
- **Rapport de référence :** [Seattle Energy Benchmarking Analysis 2016](https://seattle.gov/documents/Departments/OSE/Seattle%20Energy%20Benchmarking%20Analysis%202016%20for%20web.pdf)

Le notebook inclut un **glossaire complet** des colonnes (localisation, caractéristiques
structurelles, surfaces GFA, usages, métriques EUI, consommations par source, émissions GES,
informations administratives).

## Démarche

1. **Analyse exploratoire (EDA)** — distributions, valeurs manquantes, outliers, analyse de la cible.
2. **Filtrage en cascade (F0 → F11)** — chaque variable écartée l'est pour une raison nommée :

   | Filtre | Rôle |
   |---|---|
   | F0 | Périmètre de la mission (bâtiments non résidentiels) |
   | F1 | Colonnes sans variance (aucune information) |
   | F2 | Colonnes à fort taux de valeurs nulles ; réparation sémantique (`fillna`) plutôt qu'imputation |
   | F3 / F4 | Features PII / métadonnées / identifiants |
   | F5 | Fuites de données (mesures de la campagne 2016), **les deux cibles sont conservées** |
   | F6 | Colinéarités (`PropertyGFATotal`, `LargestPropertyUseTypeGFA`) |
   | F8 / F9 | Doublons de casse (`Neighborhood`), variable ensembliste non atomique |
   | F10 / F11 | Outliers de la cible et valeurs nulles incohérentes |

3. **Feature engineering** — encodage One-Hot des catégorielles (~180 colonnes), normalisation,
   **ablation contrôlée** `structurel strict (A)` vs `+ raccordement (B)`.
4. **Modélisation supervisée** — comparaison en validation croisée (5 plis) de familles de modèles,
   en variantes `brute` et `log1p`, évaluées en **unités réelles** (R², MAE, MedAE, MedAPE).
5. **Interprétation** — **SHAP** et **permutation importance** (dummies ré-agrégées sous leur
   variable d'origine).
6. **Ouverture** — test de **CatBoost** (catégorielles natives) et **TabPFN** (modèle de fondation).

## Résultats

Sélection **automatique** d'un modèle par cible (meilleure MedAPE sur le jeu de test) :

| Cible | Unité | Modèle retenu | Config | R² test | Erreur médiane (MedAE) | MedAPE | Biais d'inventaire |
|---|---|---|---|---|---|---|---|
| Consommation d'énergie | kBtu/an | **SVR** | B — + raccordement | 0.222 | 870 930 | 36.1 % | −39.2 % |
| Émissions de GES | t CO₂e/an | **RandomForest** | B — + raccordement | 0.086 | 24 | 47.2 % | −59.4 % |

**Apport du raccordement gaz / vapeur** (`Has_NaturalGas`, `Has_Steam`) :

- Consommation d'énergie : **+0.9 pt** de MedAPE (effet quasi nul)
- Émissions de GES : **+11.3 pts** de MedAPE (effet marqué)

**Variables décisives :** `PropertyGFABuilding(s)` > `Has_NaturalGas` > `Has_Steam`.

## Enseignements clés

- **Le levier est la source d'énergie, pas la typologie.** Un term of kBtu vaut un kBtu quelle que
  soit sa source, mais une tonne de CO₂e dépend du facteur d'émission : substituer le gaz par une
  électricité déjà décarbonée (Seattle hydraulique) pèse plus que toute intervention sur l'enveloppe.
- **Les deux cibles ne se valent pas.** La consommation se prédit depuis la structure seule ; les
  émissions n'y parviennent qu'une fois le mix énergétique connu.
- **Deux modèles livrés, pas une affirmation.** `Structurel strict` répond littéralement à la
  commande (performance plancher) ; `Enrichi` ajoute le raccordement — l'écart chiffre le retour sur
  investissement d'un partenariat avec les concessionnaires.
- **Le gain vient de la compréhension du domaine, pas de l'algorithme.** Les trois familles réglées
  convergent à quelques points de MedAPE ; le traitement natif des catégorielles (CatBoost) divise les
  colonnes par dix mais ne change ni le classement des variables ni l'écart A/B.

## Structure du dépôt

```
.
├── P3_Derue_William_template_modelistation_supervisee_data_scientist.ipynb   # Notebook complet (EDA + modélisation)
├── projet-3-seattle-energie.html                                             # Page HTML de cadrage de la mission
├── P3_DSML_Derue_William+Fiche+d'auto-évaluation_editable.pdf                # Fiche d'auto-évaluation
└── README.md
```

## Installation et exécution

```bash
# Cloner le dépôt
git clone https://github.com/WillIsback/OC-Ai-Engineer-P3.git
cd OC-Ai-Engineer-P3

# Environnement (Python 3.x)
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn shap catboost
```

Puis ouvrir le notebook :

```bash
jupyter notebook P3_Derue_William_template_modelistation_supervisee_data_scientist.ipynb
```

Le dataset est téléchargé automatiquement (`wget`) dans la première cellule. Une version
[**Google Colab**](https://colab.research.google.com/drive/1sy3lmD6vqKLnZnBXx2JGhqNAKMDvrGsE?usp=sharing)
est également disponible.

## Limites et pistes

- **Domaine de validité.** Les objets écartés le sont pour une raison nommée (anomalie de donnée ou
  de population), jamais parce qu'ils émettent beaucoup.
- **Biais d'agrégation.** La modélisation en `log` sous-estime le total du parc — le modèle est
  fiable bâtiment par bâtiment, pas pour un inventaire agrégé sans correction (facteur de Duan ou
  régression Gamma).
- **Une seule année.** Modèle calé sur 2016 ; un réentraînement annuel est nécessaire.
- **Approximation du raccordement.** `Has_NaturalGas` / `Has_Steam` ≈ « mesure > 0 » ; en production
  ils doivent venir des concessionnaires (Puget Sound Energy, Seattle Steam).

**Suites envisagées :** intervalle de prédiction (forêt quantile), évaluation de l'`ENERGYSTARScore`
(écarté par le seuil de valeurs manquantes), alternatives au `log1p` (`sample_weight`, loss Poisson
ou Gamma).

## Ressources

- [CSV 2016](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/Data_Scientist_P4/2016_Building_Energy_Benchmarking.csv)
- [Source Seattle (data.seattle.gov)](https://data.seattle.gov/Built-Environment/Building-Energy-Benchmarking-Data-2015-Present/teqw-tu6e/about_data)
- [Rapport d'analyse 2016](https://seattle.gov/documents/Departments/OSE/Seattle%20Energy%20Benchmarking%20Analysis%202016%20for%20web.pdf)
- [Notebook template OpenClassrooms](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/1047_Data+Scientist+ML/P3_DSML_1047/P3_template_modelistation_supervisee_data_scientist.ipynb)

---

*Projet réalisé dans le cadre de la formation OpenClassrooms AI Engineer — William Derue.*
