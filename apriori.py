
import pandas as pd
from apyori import apriori

################KEYWORDS IMPORT#####################
url = (r".csv")
names = names = ['ID', 'Keywords']
input = pd.read_csv(url, names=names, header=None, encoding='latin-1')
id = []
keywords = []
for index, row in input.iterrows():
    if row.Keywords != '[]':
        id.append(row.ID)
        keywords.append([x['name'] for x in eval(row.Keywords)])

df = pd.DataFrame({'ID': id, 'Keywords': keywords}) 
#print(df.shape)

################GENRES IMPORT#####################
urll = (r".csv")
names = names = ['ID', 'Genres']
inputt = pd.read_csv(urll, names = names, header=None, encoding='latin-1')
id = []
genres = []
for index, row in inputt.iterrows():
    if row.Genres != '[]':
        id.append(row.ID)
        genres.append([x['name'] for x in eval(row.Genres)])

df2 = pd.DataFrame({'ID': id, 'Genres': genres})
#print(df2.shape)

################MERGE AND CLEAN#####################
dfmerged = pd.merge(df, df2, on = 'ID')

# Drop rows with any empty cells
dfmerged.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)

dfmerged.Keywords.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
dfmerged.Genres.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)

dfmerged.drop(
    labels=0,
    axis=0,
    index=None,
    columns=None,
    level=None,
    inplace=True,
    errors='raise'
)

dfmerged = dfmerged.drop('ID', axis = 1)  
dfmerged['Keywords'] = dfmerged['Keywords'].str.join(', ')
dfmerged['Genres'] = dfmerged['Genres'].str.join(', ')

#print(dfmerged.shape)



################APRIORI#####################
records = []
for i in range(0, 31283):
   records.append([str(dfmerged.values[i,j]) for j in range(0, 2)])
    
association_rules = apriori(records, min_support=0.0025, min_confidence=0.0005, min_lift=1.01, min_length=3)
association_results = list(association_rules)

print(len(association_results))
#print((association_results[0]))

for item in association_results:

    # first index of the inner list
    # Contains base item and add item
        
    pair = item[0] 
    items = [x for x in pair]
    
    if not item[0] == 'nan' and not item[1] == 'nan':
        print("Rule: " + items[0] + " -> " + items[1])
    
        #second index of the inner list
        print("Support: " + str(item[1]))
    
        #third index of the list located at 0th
        #of the third index of the inner list
    
        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        print("=====================================")
