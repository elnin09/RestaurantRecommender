import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import word_tokenize
import joblib 





abs_tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=0, max_features=20000,
               stop_words='english', use_idf=True)

title_tfidf_vectorizer = TfidfVectorizer(max_features=20000,
               stop_words='english', use_idf=True)




df = pd.read_csv('zomato_refined.csv',nrows=10000)
#print(df)




df.dropna()
#print(df.head())
x = abs_tfidf_vectorizer.fit_transform(df['review'].values.astype('U'))
#print(df.head())





df1 = pd.DataFrame(x.toarray(), columns=abs_tfidf_vectorizer.get_feature_names())
res = pd.concat([df, df1], axis=1)

#print(res.head(10))

X = res.drop(columns=['name','url', 'rate', 'votes', 'phone', 'location', 'dish_liked', 'review'])
#print(X.head())

#print(X.columns)

y = res['name'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)








dfs = pd.read_csv('zomato.csv',usecols=['name','url','dish_liked'],nrows=25000)
#print(df)




dfs.dropna()
#print(df.head())
xs = abs_tfidf_vectorizer.fit_transform(dfs['dish_liked'].values.astype('U'))
#print(df.head())





df1s = pd.DataFrame(xs.toarray(), columns=abs_tfidf_vectorizer.get_feature_names())
ress = pd.concat([dfs, df1s], axis=1)

#print(res.head(10))

Xs= ress.drop(columns=['name','url','dish_liked'])
#print(Xs.head())
#print(Xs.columns)

ys = ress['name'].values

Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=0.33, random_state=1)







knn_reviews = joblib.load('reviews.pkl')  
knn_cuisines = joblib.load('cuisines.pkl')

while(True):
	query = input("Enter your query\n")
	if query == "":
		print("You have entered a blank response program will now exit")
		break
	query_tokens = query.split(' ');
	querydict = dict()
	for word in query_tokens:
		querydict[word]=1;

	queryframedict = dict()

	for col in X.columns: 
	    if col in querydict:
	    	queryframedict[col]=1
	    else:
	    	queryframedict[col]=0

	queryframedict2 = dict()
	for col in Xs.columns:
		if col in querydict:
			queryframedict2[col]=1
		else:
			queryframedict2[col]=0



	#print (queryframedict)   	
    
	querydataframe = pd.DataFrame([queryframedict])
	results = knn_reviews.kneighbors(querydataframe)[1].tolist()
	print("Here are your suggestionsbased on reviews")
	for resrecommend in results:
		print(y_train[resrecommend])
		print("\n")

	querydataframe = pd.DataFrame([queryframedict2])
	print("Here are your top suggestions based on dishes")
	results = knn_cuisines.kneighbors(querydataframe)[1].tolist()
	for resrecommend in results:
		print(ys_train[resrecommend])
		print("\n")
    
    
    	
    	

