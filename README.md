# Restaurant Recommender

This tool provides relevant restaurants to user based on query.

## Setup

```bash
# install pandas, sklearn, nltk,joblib
pip3 install pandas
pip3 install sklearn
pip3 install nltk
pip3 install joblib
```

## Data Source

Zomato Dataset from Kaggle is used.
https://www.kaggle.com/himanshupoddar/zomato-bangalore-restaurants

The data is cleaned and refined to filter out restaurants with less than 3.5 rating and atleast 100 votes The code for the same is present in datacleaning.py.

## Code Implementation



**Datacleaning.py** This file is used to clean our data that we received from our data set. We convert review list to individual reviews and then get zomato_refined.csv. We also remove stop words from the reviews using stop words in sklearn and nltk

**Recommender.py** - This file contains the code to train knn model from user reviews. Firstly we read the data from zomato_refined.csv and fetch all the reviews. Thereafter we calculate the tf-idf vector for each review using sklearn. After getting the tf-idf
we split our data into training and test set. After splitting our data, we train our model using sklearn. After training our model, we save this model as reviews.pkl using joblib. We can use this model later on to predict the user query. For unit testing of this model, we also take input for the prediction.

**Recommendercuisine.py**- This file is similar to Recommender.py except for the fact that we are training the model on dishes liked. This model can also be tested by entering a user query and verifying the results.

**KNeighborsClassifier** - Classifier implementing the k-nearest neighbors vote. It uses ‘feature similarity’ to predict the values of new datapoints which further means that the new data point will be assigned a value based on how closely it matches the points in the training set.
This is predefined function of sklearn.neighbors. We are basically using this classifier to fetch the k-nearest neighbors.

**Main.py**- This file can be used to test the project. We basically first load our pre-trained models using joblib and after that we create the query vector from the user input. Thereafter, we recommend users the top restaurants on the basis of user query.




**Reading the dataset** - pandas.read_csv() reads a comma-separated values (csv) file into DataFrame.



**TfidfVectorizer** - Convert a collection of raw documents to a matrix of TF-IDF features.This is predefined function of sklearn.feature_extraction.text.

## Running code

We already have trained model cuisines.pkl and reviews.pkl
In order to run the queries on already trained model, user needs to execute following command in the terminal.

``` shell
python3 main.py

#In order to train the models and then run the queries, execute following python commands

python3 recommendercuisine.py
python3 recommender.py
```


