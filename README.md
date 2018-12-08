# google-revenue-prediction
Data files:
1. train_v2.7z : original training data. Unzipping will give you train_v2.csv (24GB). It is used as input to data_cleaning.ipynb
2. test_v2.7z : original test data. Unzipping will give you test_v2.csv (7.4GB). It is used as input to data_cleaning.ipynb
3. train.csv : output of data_cleaning.ipynb. It is used as input in LinearRegression.Rmd & DecisionTree.Rmd
4. test.csv : output of data_cleaning.ipynb. It is used as input in LinearRegression.Rmd & DecisionTree.Rmd
5. trainEncoded.csv: output of data_exploration.ipynb. It is used as input in RandomForest.ipynb 
6. testEncoded.csv: output of data_exploration.ipynb. It is used as input in RandomForest.ipynb 

Cleaning & explorations:
1. data_cleaning.ipynb : Clean the data to a more usable and managable format & size. 
Input: train_v2.csv & test_v2.csv. 
Output: train.csv & test.csv.
2. data_exploration.ipynb : Exploratory visualizations and more encoding. 
Input: train.csv & test.csv. 
Output: trainEncoded.csv & testEncoded.csv
3. Explorations.Rmd : Exploratory visualizations. Input: train.csv

Predictive Models:
1. LinearRegression.Rmd : Linear Regression implementation. Input: train.csv & test.csv
2. DecisionTree.Rmd : Decision Tree implementation. 
Input: train.csv & test.csv -> preprocess_enpm.ipynb(to convert the categorical variables to factors) -> train_preprocessed.csv & test_preprocessed.csv -> DecisionTree.Rmd
3. RandomForest.ipynb : Random Forest implementation: Input: trainEncoded.csv & testEncoded.csv
4. lgbm.ipynb & lgbm.py: LightBGM implementation: Input: trainEncoded.csv & testEncoded.csv
5. mlp.ipynb & mlp.py: Neural Networks implementation: Input: trainEncoded.csv & testEncoded.csv

HTML Files:
The included HTML files are all files generated from the R markdown files and python notebooks to show outputs of each cell.
