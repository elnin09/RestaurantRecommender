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




df = pd.read_csv('zomato.csv',usecols=['name','url','dish_liked'],nrows=25000)
#print(df)




df.dropna()
#print(df.head())
x = abs_tfidf_vectorizer.fit_transform(df['dish_liked'].values.astype('U'))
#print(df.head())





df1 = pd.DataFrame(x.toarray(), columns=abs_tfidf_vectorizer.get_feature_names())
res = pd.concat([df, df1], axis=1)

#print(res.head(10))

X = res.drop(columns=['name','url','dish_liked'])
print(X.head())
print(X.columns)

y = res['name'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
knn = KNeighborsClassifier(n_neighbors = 10)


knn.fit(X_train,y_train)

joblib.dump(knn, 'cuisines.pkl') 


print(knn.score(X_test, y_test))


while(True):
	query = input("Enter your query\n")
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

	#print (queryframedict)   	

	querydataframe = pd.DataFrame([queryframedict])
	results = knn.kneighbors(querydataframe)[1].tolist()
	for resrecommend in results:
		print(y_train[resrecommend])




    	

    



