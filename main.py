from __future__ import unicode_literals
import pandas as pd
import sys
import os
from collections import OrderedDict

#scan for k 
desired_k = sys.argv[1]
desired_k = int(desired_k)

#list of all columns
names = (
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income'
)

#columns which require special attention to get median
categorical = set((
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'sex',
    'native-country',
    'race',
    'income',
))

#declare the feature and sensitive columns
feature_columns = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'race', 'sex', 'native-country']
sensitive_column = 'income'

# We load the data using Pandas
df = pd.read_csv("adult.data", sep=", ", header=None, names=names, index_col=False, engine='python');

#set columns which require special attention to catgory type ---> easier indentification and processing
for column in categorical:
    df[column] = df[column].astype('category')

#remove rows which have incomplete data for the feature or sensitive columns
for i in feature_columns:
    df = df[~df[i].isin(['?'])]
df = df[~df['income'].isin(['?'])]

#function to calculate extent/span of each column ---> required to get median
#separate handling for categorical and non-categorical features
def column_span(df, partition, scale=None):
    spans = {}
    for column in df.columns:
        if column in categorical:
            span = len(df[column][partition].unique())
        else:
            span = df[column][partition].max()-df[column][partition].min()
        if scale is not None:
            span = span/scale[column]
        spans[column] = span
    return spans

#function to divide partition based on median
#separate handling for categorical and non-categorical features
def divide(df, partition, column):
    dfp = df[column][partition]
    if column in categorical:
        values = dfp.unique()
        lv = set(values[:len(values)//2])
        rv = set(values[len(values)//2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:        
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return (dfl, dfr)

#function to check whether a partition is K-anonymous ---> length should be greater than K
def is_k_anonymous(df, partition, sensitive_column, k = desired_k):
    if len(partition) < k:
        return False
    return True

# ******** ACTUAL MONDRIAN GREEDY ALGORITHM ******** ---> partition the data greedily until no partitions possible
def mondrian(df, feature_columns, sensitive_column, scale, is_valid):
    finished_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = column_span(df[feature_columns], partition, scale)
        for column, span in sorted(spans.items(), key=lambda x:-x[1]):
            lp, rp = divide(df, partition, column)
            if not is_valid(df, lp, sensitive_column) or not is_valid(df, rp, sensitive_column):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions

#calculate initial span and start the algorithm
full_spans = column_span(df, df.index)
finished_partitions = mondrian(df, feature_columns, sensitive_column, full_spans, is_k_anonymous)

# **DEBUGGING**
"""
for partition in finished_partitions:
	print(len(partition))
"""

#function to join the entries of a column in a partition
def agg_categorical_column(series):
    x = sorted(set(series))
    return ['~'.join(x)]

#function to get the minimum entry of a column in a partition
def agg_numerical_column(series):
    return [series.min()]

#function to get the maximum entry of a column in a partition
def agg_numerical_column1(series):
    return [series.max()]

#function to build the K-anonymous dataset from partitions (considering minimum entry for non-categorical columns)
def build_K_dataset(df, partitions, feature_columns, sensitive_column, max_partitions=None):
    aggregations = {}
    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column

    rows = []
    for i, partition in enumerate(partitions):
        if max_partitions is not None and i > max_partitions:
            break
        grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
        sensitive_counts = df.loc[partition].groupby(sensitive_column).agg({sensitive_column : 'count'})
        values = grouped_columns.iloc[0].to_dict(OrderedDict)
        for sensitive_value, count in sensitive_counts[sensitive_column].items():
            if count == 0:
                continue
            values.update({
                sensitive_column : sensitive_value,
                'count' : len(finished_partitions[i])

            })
            rows.append(values.copy())
    return pd.DataFrame(rows)

#function to build the K-anonymous dataset from partitions (considering maximum entry for non-categorical columns)
def build_K_dataset1(df, partitions, feature_columns, sensitive_column, max_partitions=None):
    aggregations = {}
    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column1

    rows = []
    for i, partition in enumerate(partitions):
        if max_partitions is not None and i > max_partitions:
            break
        grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
        sensitive_counts = df.loc[partition].groupby(sensitive_column).agg({sensitive_column : 'count'})
        values = grouped_columns.iloc[0].to_dict(OrderedDict)
        for sensitive_value, count in sensitive_counts[sensitive_column].items():
            if count == 0:
                continue
            values.update({
                sensitive_column : sensitive_value,
                'count' : len(finished_partitions[i])

            })
            rows.append(values.copy())
    return pd.DataFrame(rows)

#giving command to generate the maximum and minimum entry K-anonymous datasets
df_min = build_K_dataset(df, finished_partitions, feature_columns, sensitive_column)
df_max = build_K_dataset1(df, finished_partitions, feature_columns, sensitive_column)

#save the above datasets to two separate TEMPORARY files
df_min.to_csv("adult1.out", index=False, header=False, encoding='utf-8')
df_max.to_csv("adult2.out", index=False, header=False, encoding='utf-8')

#combine minimum and maximum entry data for partitions (in the above two files) to generate a file with desired output format
line_list=[]
with open('adult.out', 'w') as file:
	with open('adult1.out', 'r') as f1, open('adult2.out', 'r') as f2:
	    for line1, line2 in zip(f1, f2):		
		row1 = line1.split(",")
		row2 = line2.split(",")
		if(int(row1[1])!=int(row2[1])):
		    row1[1] = row1[1]+"~"+row2[1]
	      	if(int(row1[5])!=int(row2[5])):
		    row1[5] = row1[5]+"~"+row2[5]
		for i in range(int(row1[8])):		
			line = str(row1[1])+", "+str(row1[0])+", "+str(row1[5])+", "+str(row1[6])+", "+str(row1[7])+", "+str(row1[3])+", "+str(row1[2])+", "+str(row1[4])+", "+str(row1[9])
			line = line.strip('\n')
			line_list.append(line)
	for line in line_list:
	    file.write(str(line)+"\n")

#calculation for discernability metric
discern=0
for partition in finished_partitions:
	discern += len(partition)**2
print(discern)

#delete the temperory files
os.remove("adult1.out")
os.remove("adult2.out")
