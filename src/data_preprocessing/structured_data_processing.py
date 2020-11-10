import pandas as pd 

def check_missing_values(table):

    has_null = table.isnull().sum()
    column_with_null = has_null[has_null > 0].keys()

    if len(column_with_null) == 0:
        print('Checking missing values : no missing values.')
        return table
    else :
        print('Checking missing values : columns %s have missing values.' % ', '.join(column_with_null))
        nb_rows_with_null = table.shape[0] - table.dropna().shape[0]
        print('\tNumber of rows w/ missing values : %d (%.2f %%)' % (
            nb_rows_with_null,
            100. * nb_rows_with_null / table.shape[0]
        ) )

        res = input('\tDrop rows with null values ? [y/N]')
        if res == 'y':
            return table.dropna()
        else :
            for col in column_with_null:
                value_to_fill = input('\tReplace null values in %s (%s) (%d missing values) with ? ' % (
                    col, 
                    table[col].dtype,
                    has_null[col]
                ))
                if table[col].dtype in ('float64', 'int64'):
                    print(value_to_fill, type(value_to_fill))
