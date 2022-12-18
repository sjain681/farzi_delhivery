import pandas as pd

farzi_pairs = pd.read_csv('farzi_pairs.csv')
farzi_pairs = farzi_pairs.drop(['fuzzy_score'], axis=1)
farzi_pairs['label'] = 1
random1 = pd.read_csv('random3.csv')
random1 = random1.drop(['index'], axis = 1)
random1['label'] = 0
random2 = pd.read_csv('random4.csv')
random2 = random2.drop(['index'], axis = 1)
random2['label'] = 0
farzi_pairs_updated = pd.concat([farzi_pairs, random1, random2])
farzi_pairs_updated = farzi_pairs_updated.reset_index()
farzi_pairs_updated = farzi_pairs_updated.drop(['index'], axis=1)
print(farzi_pairs_updated)
farzi_pairs_updated.to_csv(r'neg_examples.csv', index = False)
