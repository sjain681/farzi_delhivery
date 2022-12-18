import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def printchanges(s1, s2, dp):

    i = len(s1)
    j = len(s2)
    x = {"Change": '', "To": '', "Delete": '', "Add": ''}
# Check till the end
    while(i > 0 and j > 0):

        # If characters are same
        if s1[i - 1] == s2[j - 1]:
            i -= 1
            j -= 1

        # Replace
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            x["Change"] += s1[i - 1]
            x["To"] += s2[j - 1]
            j -= 1
            i -= 1

        # Delete
        elif dp[i][j] == dp[i - 1][j] + 1:
            x["Delete"] += s1[i - 1]
            i -= 1

        # Add
        elif dp[i][j] == dp[i][j - 1] + 1:
            x["Add"] += s2[j - 1]
            j -= 1
    return x

def editDP(s1, s2):

    len1 = len(s1)
    len2 = len(s2)
    dp = [[0 for i in range(len2 + 1)]
            for j in range(len1 + 1)]

    # Initialize by the maximum edits possible
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Compute the DP Matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):

            # If the characters are same
            # no changes required
            if s2[j - 1] == s1[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            # Minimum of three operations possible
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],
                                dp[i - 1][j - 1],
                                dp[i - 1][j])

    # Print the steps
    return printchanges(s1, s2, dp)

farzi_pairs = pd.read_csv('farzi_pairs.csv')

b = []
for i in range(len(farzi_pairs)):
    name = farzi_pairs['name'][i]
    matched_name = farzi_pairs['matched_name'][i]
    b.append(editDP(matched_name, name))

farzi_pairs['changes'] = pd.Series(b)
result = farzi_pairs.copy()

result = result.drop(['fuzzy_score'], axis = 1)

change_df = result.changes.apply(pd.Series)

result1 = pd.concat([result, change_df], axis=1)
result1 = result1.drop(['changes'], axis = 1)
print(result1)

char = ' abcdefghijklmnopqrstuvwxyz&-0123456789_'

change = {}
to = {}
delete = {}
add = {}

for i in char:
    change[i] = result1.Change.str.count(i).sum()
    to[i] = result1.To.str.count(i).sum()
    delete[i] = result1.Delete.str.count(i).sum()
    add[i] = result1.Add.str.count(i).sum()

change_df = pd.DataFrame.from_dict(change, orient='index')
to_df = pd.DataFrame.from_dict(to, orient='index')
delete_df = pd.DataFrame.from_dict(delete, orient='index')
add_df = pd.DataFrame.from_dict(add, orient='index')

char_strength = pd.concat([change_df, to_df, delete_df, add_df], axis=1)

char_strength.columns = ['Change', 'To', 'Delete', 'Add']

print(char_strength[['Change']].idxmax())
print(char_strength[['To']].idxmax())
print(char_strength[['Delete']].idxmax())
print(char_strength[['Add']].idxmax())

print(char_strength[['Change']].idxmin())
print(char_strength[['To']].idxmin())
print(char_strength[['Delete']].idxmin())
print(char_strength[['Add']].idxmin())

# get_ipython().run_line_magic('matplotlib', 'widget')

plt.rcParams['figure.figsize'] = [10, 6]
char_strength.plot(kind='bar')
plt.savefig('output.png')
plt.show()

fig = px.bar(char_strength)
fig.update_layout(barmode='group')
fig.show()

print(result1.describe())