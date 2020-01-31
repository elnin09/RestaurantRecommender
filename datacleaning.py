import csv
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction import text

csv.field_size_limit(sys.maxsize)
nltk.download('punkt')
nltk.download('stopwords')

my_stop_words = text.ENGLISH_STOP_WORDS.union(["book"])
stop_words = set(stopwords.words('english'))
print(stop_words)


def process_bio(bio):
    bio = bio.encode('ascii',errors='ignore').decode('utf-8')       #removes non-ascii characters
    bio = re.sub('\s+',' ',bio)       #repalces repeated whitespace characters with single space
    return bio

with open('zomato_refined.csv', mode='w') as csvWriter:
    fieldnames = ['name','url', 'rate', 'votes', 'phone', 'location', 'dish_liked', 'review']
    writer = csv.DictWriter(csvWriter, fieldnames=fieldnames)
    writer.writeheader()

    with open('zomato.csv') as csvFile:
        csvReader = csv.DictReader(csvFile)
        lineCnt = 0

        for row in csvReader:
            
            if(lineCnt == 10000):
                break;
            #print(row)
            url = str(row['url'])
            name = str(row['name'])
            rate = str(row['rate'])
            votes = str(row['votes'])
            phone = str(row['phone'])
            location = str(row['location'])
            dish_liked = str(row['dish_liked'])
            if(rate<'3.5/5' or int(votes)<100):
                print("foundone");
                continue;

            r = str(row['reviews_list']).strip("[]")
            m = r.split(")")
            reviews = []
            lineCnt+=1;
            for r in m:
                pattern = "RATED\\n "
                pos = r.find(pattern)
                review = r[pos + len(pattern):].strip(" ")
                review = review.lower()
                
                
                word_tokens = review.split(' ');
                filtered_sentence = []
                for w in word_tokens:
                    if w not in stop_words and w.isalpha() and w not in my_stop_words:
                        filtered_sentence.append(w)
                
                review = filtered_sentence
                #print(review)
                # url,rate,votes,phone,location,dish_liked,review
                writer.writerow({'name': name,'url': url, 'rate': rate, 'votes': votes,
                                 'phone': phone, 'location': location,
                                 'dish_liked': dish_liked, 'review': review})
        print(lineCnt)