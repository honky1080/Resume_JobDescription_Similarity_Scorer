import re
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words=set(stopwords.words('english'))

def clean_and_lemmatize(text):
  text = re.sub(r'http\S+\s*', ' ', text)
  text = re.sub(r'\S+@\S+', ' ', text)
  text=re.sub(r'[^\w\s]', '',text)
  text=re.sub(r'[^a-zA-Z0-9]',' ',text)
  tokens=nltk.word_tokenize(text)
  tokens=[token.lower() for token in tokens]
  tokens=[token for token in tokens if token not in stop_words]
  lemmatized_tokens=[lemmatizer.lemmatize(token) for token in tokens]
  return lemmatized_tokens
