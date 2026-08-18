# Orange Telecom Madagascar

## Project Overview
This repository contains the codebase, predictive models, and interactive dashboard developed during a Data Science internship at Orange Madagascar. The project uses real world telecom dataset. The data comprises ~2.9 million unique users and ~39 million transactions per quarter, and 36,633 network cells.

The goal of the project were to engineer a ETL pipeline, segment the customer base, predict Customer Lifetime Value, build a Next Best Offer recommendation engine, and deploy a dynamic web dashboard to visualise regional revenue flows and network typologies.

### Note on Data Privacy. For privacy reasons, all outputs and datasets cannot be shown or distributed in this repository. All scripts are provided for demonstration only.

## Tech Stack
- Data Engineering: polars, pandas, geopandas, shapely
- Web Dashboard: dash, dash_bootstrap_components, plotly, keplergl
- Machine Learning: scikit-learn, lifetimes (BTYD), mlxtend (FP-Growth)
- Deep Learning: pytorch, optuna, PuLP
- Similarity Search and Clustering: faiss, dtaidistance, tslearn, joblib, KMeans, TSNE

## Methodologies
Analytical Dashboard
- Deployed an interactive web dashboard for CVM and CoDMICC executive teams.
- Calculating Total Revenue, Unique Users, ARPU by Technology, and Transaction volumes.
- Built choropleth network maps, tower scatter markers, and Sankey flow diagrams.

Neural NMF and FAISS
- Trained a NonNegative Matrix Factorisation neural network using Softplus activations and negative sampling.
- Memory mapped execution across 112 NUMA node threads pinned to 56 physical cores.
- Queried latent space embeddings via faiss for hidden product affinities.
- Applied IQR thresholding to isolate underserved users, followed by KMeans clustering.

Causal AI and PSM
- Used PSM via Logistic Regression to eliminate demographic confounding variables.
- Isolated a +4,945 MGA/month causal revenue uplift attributable to mobile app adoption.
- Trained a TLearner model optimised via Qini curves to target the top 13%, ~350k users, for migration.

KPrototypes Segmentation
- Used KModes to cluster numerical usage and categorical attributes.
- Segmented 350k migration targets into 6 clusters to formulate a +1.56B MGA/month revenue strategy approved by CoDMICC.

CLV and Survival Analysis
- Applied BG/NBD models to evaluate transaction frequency/recency and track user survival probability.
- Used GammaGamma sub models to estimate 12 month CLV with a 1% monthly discount rate.
- Identified 637k users carrying 91.2B MGA in long term revenue potential, good fit for uplift by nbo.

FP-Growth/NBO
- Mined sequential purchasing patterns using mlxtend FP-Growth across all transactions.
- Deployed Next Best Offer (NBO) rules achieving up to 6.88x lift for plan upgrades.
- Filtered rules to ensure product transitions is revenue positive.

Time-Series Churn Signatures and DTW
- Calculated dynamic time gaps between consecutive transactions to find user lifecycle trajectories.
- Clustered churned sequences using Sakoe-Chiba constrained Dynamic Time Warping (DTW) and computed death signatures with DTW Barycenter Averaging (DBA).
- Elaborated a parallelised and chunked C engine across 112 cores to evaluate ~3M active users against the decay signatures.
- Combined DTW shape distance and inactivity to isolate ~195k soft churn users for win back marketing campaigns.  (made because of RFM aggregation limitations)

## Repository Structure

- app.py: Plotly Dash application for visualising regional revenue flows, network coverage, and KPIs.
- build_dataset.py: ETL pipeline script that merges user locations, plan purchases, and site references into parquet files.
- data_causal.ipynb: Customer feature engineering, VIP classification using Logistic Regression and XGBoost, Propensity Score Matching, TLearner uplift modeling, and KPrototypes segmentation.
- data_exploration.ipynb: Exploratory data analysis, geospatial antenna mapping, and checking distribution patterns.
- data_faiss_bundle_exploration.ipynb: Faiss similarity search, TSNE, and KMeans clustering to detect underserved user.
- data_month_merging.ipynb: Data preprocessing script combining multiple months of quarter user location and transaction logs.
- data_stat_analysis.ipynb: Statistical testing and data distributions.
- data_time_analysis.ipynb: BG/NBD and Gamma-Gamma survival models for CLV, FP-Growth association rules for the NBO recommendation model.
- data_nmf_bundle_creation.py: PyTorch training a NonNegative Matrix Factorization neural network.
- data_pulp_bundle_creation.py: Optimisation using PuLP for constrained bundle creation.  
