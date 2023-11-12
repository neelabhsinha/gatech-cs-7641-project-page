---
layout: default
---

# Introduction

FIFA World Cup is the most popular sports event in the world (shown below [5]).

![Importance of Football](./assets/images/intro.jpg)


We will be predicting the FIFA World Cup championship by iteratively predicting results of each game. Our prediction model will in turn benefit the following industries:
1. Sports betting
2. Data driven predictions and analysis for media
4. Decision-making for sports teams and merchandise.
3. Driving fan excitement and discussions online.

## Related Work

The problem of predicting game outcomes especially in Football(also called Soccer in NA) is usually handled as a classification problem.([4],[6]) Techniques ranging from logistic regression([4]) to RNN/Deep Learning([6]) have been employed for this task. Furthermore the problem of outcome prediction is very similar across other team based games  ([1],[3]).A comprehensive survey of the use of techniques across the various sports is shown in [3].


# Problem Definition

A tournament \\(\mathcal{T}(\boldsymbol{T},\boldsymbol{G},\boldsymbol{T_b})\\) is a set of teams \\(\boldsymbol{T}\\) participating in games \\(\boldsymbol{G}\\) (either a winner or tie (only in  group stage)) over stages \\(\boldsymbol{b} = 0,1,2,...\\) ,with a set of (\\(\boldsymbol{T_b}\\)) teams qualifying to play them. Our goal is:

1. **Outcome prediction** : \\(\forall G(T_i,T_j) \in \boldsymbol{G}\\) we predict \\(\hat{G}(T_i,T_j)\\) accurately.
2. **Group Prediction** : Given  \\(\boldsymbol{T_b}\\) we predict \\(\hat{T}_{b+1}\\) and beyond. 

The notion of "accuracy" in our case is also quantified by additional metrics like **cross-entropy** , **precision** , **recall** and **F1-score** (discussed below in the metrics section.)

For our project we handle **Group Prediction** by using **Outcome Prediction** iteratively to predict the matches within the group.

For the midsem checkpoint we will be covering **Outcome Prediction** in the report.

## Overall Pipeline

![Overall Pipeline](./assets/images/pipeline.png)


Currently we have constructed the supervised portion of the end-to-end FIFA world cup suite to aid with the "**Group Prediction**" part in the problem definition.

# Dataset
## **Sources**:
We use the datasets listed below:
- [Soccer World Cup Data (Kaggle)](https://www.kaggle.com/datasets/shilongzhuang/soccer-world-cup-challenge/){:target="_blank"} 
- [All International Matches (Kaggle)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017?select=results.csv){:target="_blank"} 
- [FIFA World Rankings (Kaggle)](https://www.kaggle.com/datasets/cashncarry/fifaworldranking){:target="_blank"} 

The dataset features are described in the following figure -
![Dataset Summary](./assets/images/dataset.png)


Of these the datasets **All International Matches** and **FIFA World Rankings** are used to train and test our Machine Learning schemes ,while the dataset **Soccer World Cup Data** is used to prepare and run tournament simulations.
## **Data Preparation**

For our supervised task the relevant datasets we are interested in are "**Match level**" data named "**InternationalMatches.csv**" and "**FifaRankings.csv**". "**InternationalMatches.csv**" contains the match outcome data for matches from 1872 to 2023 with the attributes shown in the picture above. For our task we limit ourselves to data between 2000(--Change w.r.t optimal window size) and 2022,right before the FIFA world cup . "**FifaRankings.csv**" contains the daily FIFA rankings and points of teams with additional attributes shown in the diagram above. This data goes from 1991 to 2023. However we limit ourselves to the same time frame as the match level data. 

After this we join the two datasets columnwise such that relevant attributes of the teams are placed side by side with the match data .Of these we retain the following features for our analysis:
![New Features](./assets/images/basefeatures.png)
### **Data Cleaning**
We didn't find any incomplete entries.Some country's team names have changed in the past . For the purpose of tracking their historical data we have renamed them with the present name.  
### **Feature Extraction**
For the outcome prediction task \\( \hat{G}(T_i,T_j)\\) we extract features that adequately describe each team \\(T_i\\) as well as any head-to-head relationship between them. Our guiding assumption for this is that the past performance of two teams reasonably exhibits their attributes for the match in question.([7]).In our case the past time window for analysis is 15 matches. We chose this value after trying out various combinations of time-based time windows(in years) and number of matches .
**Attach the U/inverted shaped curve where n = 15 matches becomes the extrema**
 
So  based on domain understanding we add the following features :

![New Features](./assets/images/newfeatures.png)

In our case the feature "**tournament**" is categorical and can have multiple labels and as such we perform one-hot-encoding to avoid having our classifiers assign any ordinality among these. So we split it into three features namely : "**is_friendly**" , "**is_qualifier**" and **is_tournament** exactly one of which can be 1 at a time and the rest remain 0. The "**Neutral**" attributed is renamed to "**home_match_for_home_team**" for interpretability. 
Adding all these features we end up with 25 features . 

We derive the target labels based upon the difference in "**home_goals**" and "**away_goals**" .The outcome thus can be Win for the either team or a tie (total 3 making this a multi-class classification problem.). We one-hot encode the target labels as well into the following three-labels : "**home_team_win**" , **away_team_win** and **draw** .

### Feature Selection 
For Feature selection we have implemented and tried the following :
1. Forward Feature Selection
2. PCA and
3. Feature Importance using Ensemble Learners

We compare the performance of our learning techniques with features arising from these technqiues as well as the raw features.
### **PCA**
 We perform PCA on our data both for the purposes of preliminary visualization and for Dimensionality reduction to reduce the number of features required by our classifiers. However since PCA is agnostic to target labels we monitor the effect of performing PCA on training. On one hand PCA might help us by getting rid of highly correlated features ,it can also worsen the performance if some features have some non linear predictive relation with the target labels and they get truncated. 
For our purpose we limit the number of PCA components to be enough to recover 95% (**Confirm**) of the total variance. In our case the number of PCA components thus obtained comes out to be 5 . **Graph total explained variance/related rubric vs nfeatures** 

# EDA 




# Methods

--Please rewrite as per your train of thought --


For our midterm we have performed and analysed various techniques in Supervised Classification to aid us with the "**Outcome Prediction**" problem as stated in Problem Definition. We estimate the winner/loser/match-ties using probabilities . As such we start with **Logistic Regression** which generates  a simple ,linear and efficient model for our classification problem . The model thus obtained is very interpretable.However there is an inherent assumption of the target variable being linearly dependent upon the features which need not be true for actual real-world data . 

We then employ ensemble learning methods such as : **Random Forest** and **Gradient boosting** .These are more advanced methods using Decision Trees as the fundamental model . The data is split in a way to maximize parameters such as "**Information Gain**","**Gini index**" or "**Chi-Square Index**" . Random forest is an ensemble of independent decision trees which helps with the overfitting problem inherent to decision trees .The classification is done using majority voting. Gradient boosting is another ensemble technqiue where the trees are not independent but used sequentially in a way to minimize the errors in the previous trees . Parameters to be maximised under these techniques are sensitive to both linear and nonlinear dependence of the target class on features and as such may uncover complex relationships between them. The drawback however is these models are complex and hence are computationally expensive to train.Also the models obtained lack the interpretability offered by Logisitic Regression models. **Do we use the feature importance thing for any feature selection here?**

## Semi-supervised Learning
#### Motivation 

#### Artificial Data Generation


## Supervised Learning

### Logistic Regression

#### Training
**80-20 split/PCA etc.**
**Learning Curve**
#### Tuning Parameters
**GridSearch?**


### Decision Trees
#### Training
**80-20 split/PCA etc.**
**Learning Curve**
#### Tuning Parameters
**GridSearch?/ tree depth**


### Random Forest
#### Training
**80-20 split/PCA etc.**
**Learning Curve**
#### Tuning Parameters
**GridSearch?/number of trees/each tree depth**

### Gradient Boosting
#### Training
**80-20 split/PCA etc.**
**Learning Curve**
#### Tuning Parameters
**GridSearch?/number of trees/each tree depth**
### SVM
#### Training
**80-20 split/PCA etc.**
**Learning Curve**
#### Tuning Parameters
**GridSearch**




## Tournament Simulation

**Flowchart showing the structure of tournament code?**

**Bracket Predictions?**

**One full tournament run**



# Mid-Term Results and Discussion

## Outcome Prediction
We analyse the performance of the various classification schemes on our dataset as shown below:

| Classification Scheme        | Accuracy | Precision | Recall | F-1 score |
| ---------------------------- | -------- | --------- | ------ | --------- |
| Logistic Regression(Softmax) |          |           |        |           |
| Random Forest                |          |           |        |           |
| Gradient Boosting            |          |           |        |           |


**ROC-AUC Curves?** 


**Ties**


**Semisupervised learning**

### Scopes for improvement 
If we are satisfied , let's compare with some state of the art techniques/old papers to see where we stand.


# Post-MidTerm Work
We will work on the unsupervised portion of the problem related to clustering and enhance our tournament simulations and see if the unsupervised clustering technqiues offer us some new insight that helps us calibrate our supervised classifiers better.

# Project Timeline and Responsibilities

## Contributions for the Mid-Term

| Team Member | Responsibility |
|-------------|----------------|
| Ananya Sharma |  |
| Apoorva Sinha | |
| Neelabh Sinha |  |
| Snigdha Verma |  |
| Yu- Chen Lin |  |

## Project Gantt Chart

The gantt chart covering complete timeline and responsibility distribution can be found [here](https://docs.google.com/spreadsheets/d/101ID8me3ChWkl0MzavG_UmaGsH9tkSGHOLhPi9ybc2Y/edit?usp=sharing){:target="_blank"}.

# References 
1. D. Delen, D. Cogdell, and N. Kasap. A comparative analysis of data mining methods in predicting ncaa bowl outcomes.International Journal of Forecasting, 28(2):543–552, 2012 .
2. T. Horvat and J. Job. The use of machine learning in sport outcome prediction: A review.WIREs Data Mining and Knowledge Discovery, 10(5):e1380, 2020.
3. T. Horvat, J. Job, R. Logozar, and . Livada. A data-driven machine learning algorithm for predicting the outcomes of nba games.Symmetry, 15(4), 2023.
4. D. Prasetio and D. Harlili. Predicting football match results with logistic regression.2016 International Conference On Advanced Informatics: Concepts, Theory And Application (ICAICTA), pages 1–5, 2016.
5. https://www.statista.com/chart/28766/global-reach-and-tv-viewership-of-the-fifa-world-cup
6. E. Tiwari, P. Sardar and S. Jain, "Football Match Result Prediction Using Neural Networks and Deep Learning," 2020 8th       International Conference on Reliability, Infocom Technologies and Optimization (Trends and Future Directions) (ICRITO), Noida, India, 2020, pp. 229-231, doi: 10.1109/ICRITO48877.2020.9197811.
7. Dixon, M.J. and Coles, S.G. (1997), Modelling Association Football Scores and Inefficiencies in the Football Betting Market. Journal of the Royal Statistical Society: Series C (Applied Statistics), 46: 265-280. https://doi.org/10.1111/1467-9876.00065