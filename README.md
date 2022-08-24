ElasticNet Regressor for Regression using Scikit-Learn

* elastic net
* scikit learn
* regularization
* python
* feature engine
* scikit optimize
* flask
* nginx
* gunicorn
* docker
* abalone
* auto prices
* computer activity
* heart disease
* white wine quality
* ailerons


This is an Elasticnet Regressor, a linear combination of L1 and L2 regularization, implemented using Scikit- Learn. 

The sklearn library provides with ElasticNet class with adjustable parameters $\alpha$ and l1_ratio. $\alpha$ determines how much weight is given to each of the L1 and L2 penalties. $\alpha$ = 0 corresponds to Ridge regression (L1) and $\alpha$ = 1 corresponds to Lasso regression (L2). 

For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

Preprocessing includes missing data imputation, standardization, one-hot encoding etc. For numerical variables, missing values are imputed with the mean and a binary column is added to represent 'missing' flag for missing values. For categorical variable missing values are handled using two ways: when missing values are frequent, impute them with 'missing' label and when missing values are rare, impute them with the most frequent. 

HPT includes choosing the optimal values for alpha and l1_ratio. 

The main programming language is Python. Other tools include Scikit-Learn for main algorithm, feature-engine and Scikit-Learn for preprocessing, Scikit-Learn for calculating model metrics, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service. The web service provides two endpoints- /ping for health check and /infer for predictions in real time. 