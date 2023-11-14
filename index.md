---
layout: default
---

# 1. Introduction

FIFA World Cup is the most popular sports event in the world. As shown in the image below [5], its viewership surpasses all other major sports events. With the popularity of the sport comes the importance of predictive analysis of the tournament and its matches. A lot of industries seek a good prediction at different levels for these matches for different purposes like sports betting, media and broadcast analysis, tactical decision making, driving online fan excitement. 

![Importance of Football](./assets/images/intro.jpg)

FIFA World Cup happens once in four years with 32 participating teams. First, 8 groups are created with 4 teams each. In every group, each team plays the other once. 3 points are given to the winner, 0 to the loser, and 1-1 point is shared in case of a draw. After this, top 2 teams from each group qualify for knockout stages. In knockouts, each team plays a match per stage where winner moves to the next stage and loser is eliminated, until one team ultimately wins the tournament.

In this work, we predict the FIFA World Cup matches and ideal grouping of teams for a good tournament using past match results and rankings. In summary, we use these data to generate relevant features, and then use multiple supervised techniques to predict winner of a match. Apart from real data, we also explore creating fictitious matches and use semi-supervised learning in an attempt to improve the models. Alongside match predictions, we also use unsupervised clustering techniques to create groups that can facilitate a good tournament. Through these two processes, we create an end-to-end tool that can take in participating teams, build groups, predict results of matches and ultimately, predict a complete tournament.

# 2. Related Work

The problem of predicting game outcomes especially in Football (also called Soccer in North America) is usually handled as a classification problem [4],[6]. Techniques ranging from logistic regression [4] to RNN/Deep Learning [6] have been employed for this task. Furthermore the problem of outcome prediction is very similar across other team sports [1],[3]. A comprehensive survey of the use of techniques across the various sports is described in [3].

# 3. Method Overview

## 3.1 Problem Definition

A tournament \\(\mathcal{T}(\boldsymbol{T},\boldsymbol{G},\boldsymbol{T_b})\\) is a set of teams \\(\boldsymbol{T}\\) participating in games \\(\boldsymbol{G}\\) (either a winner or tie (only in  group stage)) over stages \\(\boldsymbol{b} = 0,1,2,...\\) ,with a set of (\\(\boldsymbol{T_b}\\)) teams qualifying to play them. Our goal is:

1. **Outcome prediction** : \\(\forall G(T_i,T_j) \in \boldsymbol{G}\\) we predict \\(\hat{G}(T_i,T_j)\\) accurately.
2. **Group Prediction** : Given  \\(\boldsymbol{T_b}\\) we predict \\(\hat{T}_{b+1}\\) and beyond. 

The notion of "accuracy" in our case is also quantified by additional metrics like **cross-entropy** , **precision** , **recall** and **F1-score** (discussed below in the metrics section.)

For our project we handle **Group Prediction** by using **Outcome Prediction** iteratively to predict the matches within the group.

For the midsem checkpoint we will be covering **Outcome Prediction** in the report.

## 3.2 Overall Pipeline

![Overall Pipeline](./assets/images/pipeline.png)

# 4. Implementation Details

## 4.1 Dataset
We use the datasets listed below:
- [Soccer World Cup Data (Kaggle)](https://www.kaggle.com/datasets/shilongzhuang/soccer-world-cup-challenge/){:target="_blank"} 
- [All International Matches (Kaggle)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017?select=results.csv){:target="_blank"} 
- [FIFA World Rankings (Kaggle)](https://www.kaggle.com/datasets/cashncarry/fifaworldranking){:target="_blank"} 

The dataset features are described in the following figure -
![Dataset Summary](./assets/images/dataset.png)

Of these the datasets **All International Matches** and **FIFA World Rankings** are used to train and test our Machine Learning schemes ,while the dataset **Soccer World Cup Data** is used to prepare and run tournament simulations.

#### 4.1.1 Data Cleaning

While we initially began with data from various types of matches, including individual matches in FIFA World Cups, qualifiers, friendlies and others. However, this did not yield favorable accuracy in prediction. Thus, based on our domain knowledge of the tournament, we reduced the dataset to consider only including individual matches in FIFA World Cups and qualifiers. This yielded better results across all employed methods.
#### 4.1.2 Feature Extraction
To predict the outcomes, we first extract features for a match fixture using domain knowledge and correlation analysis. For the two teams playing, we take last $n_{ind}$ individual matches in FIFA World Cups and qualifiers against any team. From this, we extract number of wins, goals scored (mean, std), goals conceded (mean, std), mean of rank difference of this team against oppositions played for each team. Alongside this, we also take in the current rank of the teams. After this, we take last $n_{h2h}$ matches against each other in the same category and extract difference in rank of the teams and mean, std of goals scored by both the teams. We also take a categorical variable of whether the match is at a neutral venue, and if it is a world cup match or a qualifier. Complete set of features are described in the table below. To get the labels, we compare the goals scored for both teams in the match and if home_team scores more, we make the label = 1, otherwise 0.

![New Features](./assets/images/basefeatures.png)

#### 4.1.3 Exploratory Data Analysis

## 4.2 Models

Using features extracted above, we train build a binary classifier using various algorithms. To start, we implement Logistic Regression [X], Support Vector Machines [X], Decision Tree [X] which are simple, efficient and interpretable algorithms and them move to ensemble methods like , Random Forest [X], and Gradient Boost [X] to predict the probability of team A winning the match. In working with the classifier, we also experiment with forward feature selection [X] to select best features from the initial feature set, and also do Principal Component Analysis [X] to reduce the dimensionality of features. We tune all these methods by defining a search space and using Randomized Search using k-fold cross validation.

As we realize that the number of data can also be a cause of concern since World Cups happen once every four years in a space of two months, we generate artificial permutation of matches of two teams. To do this, we take a date $D$ and team playing a match on that day $T_D$. Then, for each team $T_D^i$ in this set, if the team has played against a set $T_R$ teams in the past, we generate a match between $T_D^i$ and each member of set $T_R - T_D$. After this, we select a random $N_A$ set of matches from this and follow a semi-supervised learning [X] approach to train the classifier using labeled real matches and this unlabeled artificial matches to predict the results.



### 4.2.1 Hyperparameters

In all the learning algorithms employed we have a fixed set of hyperparameters (example penalty and 'c' for logistic regression, number of trees/tree depth/sampling rate for Random Forest etc). In order to find these we use **RandomizedSearchCV** followed by **GridSearchCV**. Since this is a multivariate optimization problem, randomly sampling the parameters helps us narrow down the search space. We begin with a RandomizedSearchCV in order to get to the vicinity of hyperparameters. Then, we conduct **GridSearch** in the proximity of the best performing solution of **RandomizedSearchCV** to fine tune a better performing set of hyperparameters. This helps to reduce the computational cost and complexity of a full grid search, while increasing the accuracy around a **RandomizedSearchCV**.


### 4.2.2 Model Training

TODO: TBA
# 5 Experiments

## 5.1 Supervised Model Performance

We analyze the performance of the various classification schemes on our dataset as shown below:

| Technique          | Accuracy | Precision | Recall | F-1 score | ROC-AUC |
| ------------------ | -------- | --------- | ------ | --------- | ------- |
| LogisticRegression | 0.73     | 0.73      | 0.73   | 0.73      | 0.19    |
| DecisionTree       | 0.70     | 0.71      | 0.70   | 0.70      | 0.24    |
| RandomForest       | 0.72     | 0.72      | 0.72   | 0.72      | 0.21    |
| GradientBoosting   | 0.72     | 0.73      | 0.72   | 0.72      | 0.20    |
| SVM                | 0.73     | 0.73      | 0.73   | 0.73      | 0.20    |

### 5.1.1 Confusion Matrix
<p>
  <img src="./assets/images/confusion_matrix_lr.png" alt="LogisticRegressionCM" width="350"/>
  <img src="./assets/images/confusion_matrix_dt.png" alt="DTCM" width="350"/>
  <img src="./assets/images/confusion_matrix_rf.png" alt="RFCM" width="350"/>
 </p>
 <p>
  <img src="./assets/images/confusion_matrix_gb.png" alt="GBCM" width="350"/>
  <img src="./assets/images/confusion_matrix_svm.png" alt="SVMCM" width="350"/>
 </p>


### 5.1.2 Learning Curve

<p>
  <img src="./assets/images/learning_curve_logistic_regression.png" alt="LogisticRegressionCurve" width="350"/>
  <img src="./assets/images/learning_curve_dt.png" alt="DTCurve" width="350"/>
  <img src="./assets/images/learning_curve_rf.png" alt="RFCurve" width="350"/>
 </p>
 <p>
  <img src="./assets/images/learning_curve_gb.png" alt="GBCurve" width="350"/>
  <img src="./assets/images/learning_curve_svm.png" alt="SVMCurve" width="350"/>
 </p>


### 5.1.3 ROC/AUC Curve

<p>
  <img src="./assets/images/roc_curve_lr.png" alt="LogisticRegressionROC" width="350"/>
  <img src="./assets/images/roc_curve_dt.png" alt="DTROC" width="350"/>
  <img src="./assets/images/roc_curve_rf.png" alt="RFROC" width="350"/>
 </p>
 <p>
  <img src="./assets/images/roc_curve_gb.png" alt="GBROC" width="350"/>
  <img src="./assets/images/roc_curve_svm.png" alt="SVMROC" width="350"/>
 </p>

## 5.2 Impact of Forward Feature Selection

Forward feature selection is the iterative addition of features to the model one at a time. The process starts with an empty set of features and gradually incorporates the most relevant features based on certain criteria, in our case the increase in accuracy of the model based on the set of features being added. Post forward feature selection, we found the accuracy of each model to be drop by 5%. Due to this, we did not move forward with employing this technique. A possible hypothesis and explanation for this behavior, is that individual features had lesser contribution to the accuracy of the model, and were enforced by other features of the dataset, thus leading to better accuracy without forward feature selection.

## 5.3 Impact of Principal Component Analysis

## 5.4 Impact of Semi-supervised Learning
#### Motivation 
**Show the non-converging learning curve**
#### Artificial Data Generation



## 5.5 Tournament Simulation

### 5.5.1 Tournament Schedule

We are following the official FIFA World Cup match scheduling strategy. For this simulation, we have used the official FIFA World Cup 2022 Groups.
The groups are as follows:
### 
    Group A= ['Qatar', 'Ecuador', 'Senegal', 'Netherlands']
    Group B= ['England', 'Iran', 'USA', 'Wales']
    Group C= ['Argentina', 'Saudi Arabia', 'Mexico', 'Poland']
    Group D= ['France', 'Australia', 'Denmark', 'Tunisia']
    Group E= ['Spain', 'Costa Rica', 'Germany', 'Japan']
    Group F= ['Belgium', 'Canada', 'Morocco', 'Croatia']
    Group G= ['Brazil', 'Serbia', 'Switzerland', 'Cameroon']
    Group H= ['Portugal', 'Ghana', 'Uruguay', 'South Korea']
##### Total matches= 64
#### A.  Group Stage- 8 groups of 4 teams each<br>
     Each team plays 3 matches with the other teams in the group
     Total matches per group= 6 (4C2)
     Total matches= 48
#### B. Knockout Stages- Played after Group Stages
##### 1. Round of 16- 8 Matches (16C2 Matches)<br>
     First-place Group A vs. Second-place Group B- W1
     First-place Group B vs. Second-place Group A- W2
     First-place Group C vs. Second-place Group D- W3
     First-place Group D vs. Second-place Group C- W4
     First-place Group E vs. Second-place Group F- W5
     First-place Group F vs. Second-place Group E- W6
     First-place Group G vs. Second-place Group H- W7
     First-place Group H vs. Second-place Group G- W8
 
##### 2. Quarter Finals- 4 Matches (8C2)<br>
     W1 vs W2- QF_W1
     W3 vs W4- QF_W2
     W5 vs W6- QF_W3
     W7 vs W8- QF_W4
 
##### 3. Semi Finals- 2 Matches (4C2)<br>
     QF_W1 vs QF_W2- SF_W1
     QF_W3 vs QF_W4- SF_W2
 
##### 4. Play-offs/ Third Place- 1 Match (2C2)<br> 
     Semi Final Losers
 
##### 5. Final- 1 Match<br>
     SF_W1 vs SF_W2


### Simulation
We have analysed the results using 5 different models:

![Decision Tree Simulation](./assets/images/simulation_decision_tree.png)
![Gradient Boost Simulation](./assets/images/simulation_gradient_boost.png)
![Logistic Regression Simulation](./assets/images/simulation_logistic_regression.png)
![Random Forest Simulation](./assets/images/simulation_random_forest.png)
![Support Vector Machine Simulation](./assets/images/simulation_support_vector_machine.png)

<!-- ----To see further (Neelabh's checkpoint) ------ -->

# 6 Scopes for improvement 

**Let's discuss this first once all data is on report**

# 7 Post-MidTerm Work
We will work on the unsupervised portion of the problem related to clustering and enhance our tournament simulations and see if the unsupervised clustering technqiues offer us some new insight that helps us calibrate our supervised classifiers better.
**Put the clustering discussion here if you wanna copy proposal stuff**
# 8 Project Timeline and Responsibilities

## 8.1 Contributions for the Mid-Term

| Team Member | Responsibility |
|-------------|----------------|
| Ananya Sharma |  |
| Apoorva Sinha | |
| Neelabh Sinha |  |
| Snigdha Verma |  |
| Yu- Chen Lin |  |

## 8.2 Project Gantt Chart

The gantt chart covering complete timeline and responsibility distribution can be found [here](https://docs.google.com/spreadsheets/d/101ID8me3ChWkl0MzavG_UmaGsH9tkSGHOLhPi9ybc2Y/edit?usp=sharing){:target="_blank"}.

# 9 References 
1. D. Delen, D. Cogdell, and N. Kasap. A comparative analysis of data mining methods in predicting ncaa bowl outcomes.International Journal of Forecasting, 28(2):543–552, 2012 .
2. T. Horvat and J. Job. The use of machine learning in sport outcome prediction: A review.WIREs Data Mining and Knowledge Discovery, 10(5):e1380, 2020.
3. T. Horvat, J. Job, R. Logozar, and . Livada. A data-driven machine learning algorithm for predicting the outcomes of nba games.Symmetry, 15(4), 2023.
4. D. Prasetio and D. Harlili. Predicting football match results with logistic regression.2016 International Conference On Advanced Informatics: Concepts, Theory And Application (ICAICTA), pages 1–5, 2016.
5. https://www.statista.com/chart/28766/global-reach-and-tv-viewership-of-the-fifa-world-cup
6. E. Tiwari, P. Sardar and S. Jain, "Football Match Result Prediction Using Neural Networks and Deep Learning," 2020 8th       International Conference on Reliability, Infocom Technologies and Optimization (Trends and Future Directions) (ICRITO), Noida, India, 2020, pp. 229-231, doi: 10.1109/ICRITO48877.2020.9197811.
7. Dixon, M.J. and Coles, S.G. (1997), Modelling Association Football Scores and Inefficiencies in the Football Betting Market. Journal of the Royal Statistical Society: Series C (Applied Statistics), 46: 265-280. https://doi.org/10.1111/1467-9876.00065