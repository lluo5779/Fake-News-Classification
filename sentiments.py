from textblob import TextBlob
# Each word in the lexicon has scores for:
# 1)     polarity: negative vs. positive    (-1.0 => +1.0)
# 2) subjectivity: objective vs. subjective (+0.0 => +1.0)

def calc_text_polarity(df):
  ids = []
  polarity = []
  output = pd.DataFrame()
  for id in df.id.unique():
    texts = df.loc[df['id'] == id, ['text']]
    ids.append(id)
    polarity.append(TextBlob(' '.join(str(x) for x in texts.text)).sentiment[0])
  output['id'] = ids
  output['polarity'] = polarity
  return output

def calc_text_subjectivity(df):
  ids = []
  subjectivity = []
  output = pd.DataFrame()
  for id in df.id.unique():
    texts = df.loc[df['id'] == id, ['text']]
    ids.append(id)
    subjectivity.append(TextBlob(' '.join(str(x) for x in texts.text)).sentiment[1])
  output['id'] = ids
  output['subjectivity'] = subjectivity
  return output
