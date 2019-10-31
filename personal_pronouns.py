def pronoun_list():
  personal_pronouns = ["I", "me", "mine", "my", "you", "yours", "your", "we", "us", "our", "ours"]
  return personal_pronouns

def count_pronouns(text, pronouns):
  if type(text) != str:
    return 0
  total = 0
  for pronoun in pronouns:
    total += text.count(pronoun)
  return total

def create_pronoun_counts(stage_three_text):
  pronouns = pronoun_list()
  pronoun_counts = pd.DataFrame()
  ids = []
  counts = []
  for id in stage_three_text.id.unique():
    texts = stage_three_text.loc[stage_three_text['id'] == id, ['text']]
    count = 0
    for snippet in texts.text:
      count += count_pronouns(snippet, pronouns)
    ids.append(id)
    counts.append(count)
  pronoun_counts['id'] = ids
  pronoun_counts['count'] = counts
  return pronoun_counts