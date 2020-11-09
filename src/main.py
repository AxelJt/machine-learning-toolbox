from tree_models.decisiontree import SuperDTC
import pandas as pd 

# -- import data -- #
table = pd.read_csv('requests_customers.csv')

# -- pre-process data -- #

# ------- check null values -------- #
has_null = table.isnull().sum()

if (has_null > 0).sum() == 0:
    print('No NULL values in dataset.')
else:
    print(
        'Columns w/ NULL values :'+', '.join(has_null[has_null > 0].keys())
    )

for col in has_null[has_null > 0].keys():
    table[col] = table[col].fillna('unknown')

# ------- get dummies -------- #

ids_column = 'trip_form_id'
target = 'is_sold'
column_to_dummy = list(table)
column_to_dummy.remove(ids_column)
column_to_dummy.remove(target)

column_to_dummy.remove('has_description')
column_to_dummy.remove('date_known')


cleaned_table = pd.get_dummies(table, columns=column_to_dummy, drop_first=False)

# -- train models -- #

features = list(cleaned_table)
features.remove(ids_column)
features.remove(target)

clf = SuperDTC(
    X_train=cleaned_table[features].values, 
    X_test=cleaned_table[features].values, 
    y_train=cleaned_table[target].values, 
    y_test=cleaned_table[target].values, 
    params=dict(max_depth=5, min_samples_leaf=.05)
)

clf.fit()

# tree_dict = clf.clusters_descr(features)
# for k, v in tree_dict.items():
#     print("\nCluster", k)
#     print(v)
    
clf.tree_to_code(features)