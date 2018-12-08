# google-revenue-prediction
1. Data Collection
    Solving https://www.kaggle.com/c/ga-customer-revenue-prediction
    Original file: train_v2.csv(~24GB)

2. Data Cleaning -> data_cleaning.ipynb
    1. The original CSV file is read as chunks.
    2. JSON columns were flattened into CSV.
    3. Columns with constant values were removed.
    4. Unstructured JSON columns such as hits was removed from the data set
    5. Setting 'unknown.unknown', '(not set)', 'not available in demo dataset', '(not provided)', '(none)', '<NA>' to 'NA'
    6. Output train.csv(~460MB)
  
3. Data Exploration -> data_exploration.ipynb and Explorations.Rmd
    1. Data Exploratory Visualizations
    
4. Feature Engineering -> data_exploration.ipynb
    1. Day, Month, Year added
    2. Encoding (WOE, Label)
    3. Setting ‘NA’s of transactionRevenue(target variable) to zero.
    4. Output trainEncoded.csv(~260MB)
    
5. Predictive Modelling
    1. Linear Regression -> LinearRegression.Rmd
    2. DT -> To covert categorical variables to factors and to add new variables -> preprocess_enpm.ipynb 
                     Output: train_preprocessed.csv, test_preprocessed.csv   
                     DecisionTree.Rmd(Run Decision tree using train_preprocessed.csv, test_preprocessed.csv)
    2. Random Forest -> RandomForest.ipynb
    3. LGBM -> lgbm.ipynb & lgbm.py
    4. NN -> mlp.ipynb & mlp.py
