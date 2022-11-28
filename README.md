ElasticNet Regressor using Scikit-Learn

- elastic net
- scikit learn
- regularization
- python
- feature engine
- scikit optimize
- flask
- nginx
- gunicorn
- docker

This is an Elasticnet Regressor, a linear combination of L1 and L2 regularization, implemented using Scikit- Learn.

The sklearn library provides with ElasticNet class with adjustable parameters $\alpha$ and l1_ratio. $\alpha$ determines how much weight is given to each of the L1 and L2 penalties. $\alpha$ = 0 corresponds to Ridge regression (L1) and $\alpha$ = 1 corresponds to Lasso regression (L2).

For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

The data preprocessing step includes:

- for categorical variables
  - Handle missing values in categorical:
    - When missing values are frequent, then impute with 'missing' label
    - When missing values are rare, then impute with most frequent
- Group rare labels to reduce number of categories
- One hot encode categorical variables

- for numerical variables

  - Add binary column to represent 'missing' flag for missing values
  - Impute missing values with mean of non-missing
  - MinMax scale variables prior to yeo-johnson transformation
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale data after yeo-johnson

- for target variable
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale target data after yeo-johnson

HPT includes choosing the optimal values for alpha and l1_ratio.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as abalone, ailerons, auto_prices, computer_activity, diamond, energy, heart_disease, house_prices, medical_costs, and white_wine.

The main programming language is Python. Other tools include Scikit-Learn for main algorithm, feature-engine and Scikit-Learn for preprocessing, Scikit-Learn for calculating model metrics, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service. The web service provides two endpoints- /ping for health check and /infer for predictions in real time.
