---
layout: default
---

# Introduction

FIFA World Cup is the most popular sports event in the world (shown below [5]).

![Importance of Football](/gatech-cs-7641-project-page/assets/images/intro.jpg)
 
We will be predicting the FIFA World Cup championship by iteratively predicting results of each game. Our prediction model will in turn benefit the following industries:
1. Sports betting
2. Data driven predictions and analysis for media
4. Decision-making for sports teams and merchandise.
3. Driving fan excitement and discussions online

# Problem Definition

A tournament \\(\mathcal{T}(\boldsymbol{T},\boldsymbol{G},\boldsymbol{T_b})\\) is a set of teams \\(\boldsymbol{T}\\) participating in games \\(\boldsymbol{G}\\) (either a winner or tie (only in  group stage)) over stages \\(\boldsymbol{b} = 0,1,2,...\\) ,with a set of (\\(\boldsymbol{T_b}\\)) teams qualifying to play them. Our goal is:

1. **Outcome prediction** : \\(\forall G(T_i,T_j) \in \boldsymbol{G}\\) we predict \\(\hat{G}(T_i,T_j)\\) accurately.
2. **Group Prediction** : Given  \\(\boldsymbol{T_b}\\) we predict \\(\hat{T}_{b+1}\\) and beyond. 

In literature([1],[2],[3]) this is usually modelled as a classification problem ,but there are exceptions([4]). 


# Dataset
## **Sources**:
- [Soccer World Cup Data (Kaggle)](https://www.kaggle.com/datasets/shilongzhuang/soccer-world-cup-challenge/){:target="_blank"} 
- [All International Matches (Kaggle)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017?select=results.csv){:target="_blank"} 
- [FIFA World Rankings (Kaggle)](https://www.kaggle.com/datasets/cashncarry/fifaworldranking){:target="_blank"} 

The dataset features are described in the following figure -
![Dataset Summary](/gatech-cs-7641-project-page/assets/images/dataset.png)


## **Usage**:
1. Clubbing teams into different ‘groups’ based on relevant extracted  features.
2. Combining these with FIFA rankings and historical data to predict the winner.

# Methods

## Supervised Methods (for match outcome prediction)

We propose using classification algorithms for Outcome Prediction. Given a match fixture, we extract relevant features and use it to predict the teams' winning probabilities using:

- Logistic Regression: Simple, linear, but efficient and more interpretable model that can yield predictions assuming linear dependency on features.
- Random Forest: Advanced, more expensive and less interpretable, but can capture complex relationships effectively hence allowing us to study the trade-off between factors.

## Unsupervised Methods (for grouping)

We create ideal groups (A, B, C...) from the set of teams for fair competition. The fixed number of FIFA groups fixes the number of clusters, hence requiring clustering methods with a known initial number of clusters.
The following shall be explored -
- KMeans (Hard Clustering): Simple, efficient and more interpretable method to create groups
- Gaussian Mixture Model (Soft Clustering): Computationally expensive, but with flexible cluster shapes and responsibilities which can provide more insights on the groupings

In addition, dimensionality reduction will be used to preprocess all unsupervised and supervised models for effective feature selection.


## Overall Pipeline

![Overall Pipeline](/gatech-cs-7641-project-page/assets/images/pipeline.png)

We propose this end-to-end FIFA world cup suite where with the given historical and ranking data, we generate groups using clustering, and use it to build tournament compatible match schedules. The supervised classifiers then predict all matches iteratively through the tournament until the winner is decided.

# Potential Results and Discussion

A thorough study comparing the models’ performance, efficiency and robustness is planned. 

Classification methods’  metrics:
- cross-entropy loss (match-level, tournament stage level, and overall)
- accuracy
- precision 
- recall
- F1 score
- ROC-AUC

 Clustering algorithms’ metrics:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

We are using internal measures only, as we care only about the quality of groups generated rather than comparing to any ground truth (which need not be fixed in our use case).

# Project Timeline and Responsibilities

## Contributions for the Proposal

| Team Member | Responsibility |
|-------------|----------------|
| Ananya Sharma | Ideation, Dataset Extraction and Unsupervised methods |
| Apoorva Sinha | Ideation, Problem Definition and Expected Results |
| Neelabh Sinha | Ideation, Supervised Methods and Expected Results |
| Snigdha Verma | Ideation, Dataset Extraction and Unsupervised methods |
| Yu- Chen Lin | Ideation, Introduction, Background and Literature Survey |

## Project Gantt Chart

The gantt chart covering complete timeline and responsibility distribution can be found [here](https://docs.google.com/spreadsheets/d/101ID8me3ChWkl0MzavG_UmaGsH9tkSGHOLhPi9ybc2Y/edit?usp=sharing){:target="_blank"}.

# References 
1. D. Delen, D. Cogdell, and N. Kasap. A comparative analysis of data mining methods in predicting ncaa bowl outcomes.International Journal of Forecasting, 28(2):543–552, 2012 .
2. T. Horvat and J. Job. The use of machine learning in sport outcome prediction: A review.WIREs Data Mining and Knowledge Discovery, 10(5):e1380, 2020.
3. T. Horvat, J. Job, R. Logozar, and . Livada. A data-driven machine learning algorithm for predicting the outcomes of nba games.Symmetry, 15(4), 2023.
4. D. Prasetio and D. Harlili. Predicting football match results with logistic regression.2016 International Conference On Advanced Informatics: Concepts, Theory And Application (ICAICTA), pages 1–5, 2016.
5. https://www.statista.com/chart/28766/global-reach-and-tv-viewership-of-the-fifa-world-cup

