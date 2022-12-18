import pandas as pd

file = pd.read_csv("farzi_pairs.csv")
practice = file[['name', 'matched_name']]

practice1 = practice['matched_name'].sample(frac=1).reset_index()
practice2 = pd.concat([practice['name'], practice1], axis=1)

for i in range(len(practice2)):
    if practice2['index'][i] == i:
        print(True, i)

practice2.to_csv(r'random3.csv', index = False)

practice3 = practice['matched_name'].sample(frac=1).reset_index()
practice4 = pd.concat([practice['name'], practice3], axis=1)

for i in range(len(practice4)):
    if practice2['index'][i] == i:
        print(True, i)

practice4.to_csv(r'random4.csv', index = False)