# Use a pipeline as a high-level helper
from transformers import pipeline

# classify = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
classify = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

dataset = pd.read_csv("SPAM Data.csv")
dataset

classify("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's")

reviews = list(dataset['Message'])
sentiments = []
for r in tqdm(reviews):
  if classify(r)[0]['label'] == 'LABEL_1':
    sentiments.append("spam")
  else:
    sentiments.append("ham")

y = dataset['Category']
accuracy_score(y, sentiments)