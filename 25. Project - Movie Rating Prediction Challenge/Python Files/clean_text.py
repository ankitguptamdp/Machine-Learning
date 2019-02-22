from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sys # For reading command line argument

sample_text="""I loved this movie since I was 7 and I saw it on the opening day. It was so touching and beautiful. I strongly recommend seeing for all. It's a movie to watch with your family by far.<br /><br />My MPAA rating: PG-13 for thematic elements, prolonged scenes of disastor, nudity/sexuality and some language."""

# Initialise Objects
tokenizer=RegexpTokenizer(r'\w+') 
# r stands for Regular Expression 
# w stands for all the words
en_stopwords=set(stopwords.words('english'))
ps=PorterStemmer()

def getCleanedReview(review):
    review=review.lower()
    review=review.replace('<br /><br />',' ')    
    # Tokenize
    tokens=tokenizer.tokenize(review)
    # Stopwords removal
    new_tokens=[token for token in tokens if token not in en_stopwords]
    # Stemming
    stemmed_tokens=[ps.stem(token) for token in new_tokens]    
    cleaned_review=' '.join(stemmed_tokens)    
    return cleaned_review

# Write one function that accepts input file and returns clean output file of movies
def getStemmedDocument(inputFile,outputFile):
	out=open(outputFile,'w',encoding='utf8')
	with open(inputFile,encoding='utf8') as f: # In windows sometimes it creates problem (encoding issues) if encoding is not mentioned seperately
		reviews=f.readlines() # Read reviews line by line
	for review in reviews:
		cleaned_review=getCleanedReview(review)
		print((cleaned_review),file=out)
	out.close()

# Read command line arguments
#inputFile=sys.argv[1]
#outputFile=sys.argv[2]
#getStemmedDocument(inputFile,outputFile)

# Command in teminal
# python3 clean_reviews.py ../Text\ Files/imdb_toy_x.txt ../Text\ Files/imdb_toy_clean.txt
