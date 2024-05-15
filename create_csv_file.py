import csv

# Sample data
comments = [" India faces challenge bcos 1.Size population 2.Population density 3.Stage development&amp;thus fundAvailability(it's nationBUTnotAsAdvancedAsChinaSK,Taiwan form govt vsChinas authoritarianSystem makes admin certainMeasures difficult	True", "he is disgusting mexican"," Why do I smoke weed still? Because I work with some people who are not accountable for their actions and saying I started just because I'm fucking Asian American...I try not to let it affect me but it does...when it has taken countless lives... FUCK YOU",
                   "25 jan. interested in how #discrimination and  has affected the criminal justice system?"]
labels = ['xenophobia', 'racism', 'racism', 'racism']

data = zip(comments, labels)
csv_file = 'comments_labels.csv'

with open(csv_file, 'w', newline='' ) as file:
    writer = csv.writer(file)
    writer.writerow(['Comments', 'Labels'])
    writer.writerow(data)

print(f"CSV file '{csv_file}' has been created successfully.")
