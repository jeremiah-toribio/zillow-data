# ==--==--== Imports ==--==--==
import env
import os
# Ignore Warning
import warnings
warnings.filterwarnings("ignore")
# Array and Dataframes
import numpy as np
import pandas as pd
# Imputer
from sklearn.impute import SimpleImputer
# Evaluation: Visualization
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# Evaluation: Statistical Analysis
from scipy import stats
# Modeling: Scaling
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
# Modeling
from sklearn.model_selection import GridSearchCV
# Metrics

# fips = county
#6037 - LA
#6059 - orange
#6111 - Ventura

# **************** FUNCTION TO GET SQL CONNECTION *****************
# Create helper function to get the necessary connection url.


def zillow(database='zillow',user=env.user, password=env.password, host=env.host):
    '''
    Pulls data from mySql server and drops duplicate columns and values (keeps 1 of needed)
    encodes all categorical data and drops columns that are unnecessary as a by product of 
    new dummy columns.
    '''
    if os.path.isfile('zillow.csv'):
        # If csv file exists read in data from csv file.
        print('File exists, pulling from system.')
        zillow = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        # pulling data from mysql
        print('No file exists, extracting from MySQL.')

        query = '''SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt,
                yearbuilt, taxamount, fips, propertylandusetypeid, transactiondate
                FROM properties_2017 as p
                LEFT JOIN predictions_2017 as pd
                ON p.id = pd.id
                WHERE propertylandusetypeid = 261'''
        connection = f'mysql+pymysql://{user}:{password}@{host}/{database}'
        zillow = pd.read_sql(query, connection)

        # Dropping of bedroomcnt, bathroomcnt values where they are 0
        zillow['bedroomcnt'][zillow['bedroomcnt'] == 0].replace(0, np.nan,inplace=True)
        zillow['bathroomcnt'][zillow['bathroomcnt'] == 0].replace(0, np.nan, inplace=True)
        # Null values determined to not be worth altering as all are > 1%
        zillow = zillow.dropna()

        
        # rename columns
        zillow = zillow.rename(columns={'fips':'county', 'taxvaluedollarcnt':'tax_value','calculatedfinishedsquarefeet':'sq_feet'})
        zillow['county'] = zillow['county'].map({6037:'LA',6059:'Orange',6111:'Ventura'})

        # changing value types
        zillow['yearbuilt'] = zillow['yearbuilt'].astype(int)
        zillow['bedroomcnt'] = zillow['bedroomcnt'].astype(int)
        zillow['sq_feet'] = zillow['sq_feet'].astype(int)
        zillow['tax_value'] = zillow['tax_value'].astype(int)
        
        # removing duplicate columns
        zillow = zillow.loc[:,~zillow.columns.duplicated()].copy()
        
        # normalizing numerical data
        # encode categorical data
        dummies = pd.get_dummies(zillow['county'],dtype=int)
        zillow = pd.concat([zillow, dummies], axis=1)
        # EXAMPLE: telco['total_charges'] = telco['total_charges'].str.replace(' ','0').astype('float')

        # additional columns for visualization
        
        # engineered features

        
        
        # dropping extra columns
        zillow = zillow.drop(columns=['propertylandusetypeid','taxamount','transactiondate'])
        # restoring 'drop_first' column for contract_type as it is desired to specify just this value type (without deducting)

        # lowering all column names
        zillow.columns = map(str.lower,zillow.columns)
        zillow.columns = zillow.columns.str.replace(' ','_')

        # cache data
        zillow.to_csv('zillow.csv')
    return zillow

def zillow_post(database='zillow',user=env.user, password=env.password, host=env.host):
    '''
    Pulls data from mySql server and drops duplicate columns and values (keeps 1 of needed)
    encodes all categorical data and drops columns that are unnecessary as a by product of 
    new dummy columns.
    '''
    if os.path.isfile('zillow.csv'):
        # If csv file exists read in data from csv file.
        print('File exists, pulling from system.')
        zillow = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        # pulling data from mysql
        print('No file exists, extracting from MySQL.')

        query = '''SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt,
                yearbuilt, taxamount, fips, propertylandusetypeid, transactiondate, garagetotalsqft, fireplacecnt
                FROM properties_2017 as p
                LEFT JOIN predictions_2017 as pd
                ON p.id = pd.id
                WHERE propertylandusetypeid = 261;'''
        connection = f'mysql+pymysql://{user}:{password}@{host}/{database}'
        zillow = pd.read_sql(query, connection)

        # Null values determined to not be worth altering as all are > 1%
        zillow = zillow.dropna()

        # rename columns
        zillow = zillow.rename(columns={'fips':'county', 'taxvaluedollarcnt':'tax_value','calculatedfinishedsquarefeet':'sq_feet'})
        zillow['county'] = zillow['county'].map({6037:'LA',6059:'Orange',6111:'Ventura'})

        # changing value types
        zillow['yearbuilt'] = zillow['yearbuilt'].astype(int)
        zillow['bedroomcnt'] = zillow['bedroomcnt'].astype(int)
        zillow['sq_feet'] = zillow['sq_feet'].astype(int)
        zillow['tax_value'] = zillow['tax_value'].astype(int)
        
        # removing duplicate columns
        zillow = zillow.loc[:,~zillow.columns.duplicated()].copy()
        
        # normalizing numerical data
        # encode categorical data
        dummies = pd.get_dummies(zillow['county'],dtype=int)
        zillow = pd.concat([zillow, dummies], axis=1)
        # EXAMPLE: telco['total_charges'] = telco['total_charges'].str.replace(' ','0').astype('float')

        # additional columns for visualization
        
        # engineered features

        
        
        # dropping extra columns
        zillow = zillow.drop(columns=['propertylandusetypeid','taxamount','transactiondate'])
        # restoring 'drop_first' column for contract_type as it is desired to specify just this value type (without deducting)

        # lowering all column names
        zillow.columns = map(str.lower,zillow.columns)
        zillow.columns = zillow.columns.str.replace(' ','_')

        # cache data
        zillow.to_csv('zillow.csv')
    return zillow
    
### To get single family homes, no query ###
    #single_family = zillow[zillow['propertylandusetypeid'] == 261]
    #single_family

def splitter(df,target='tax_value', stratify=None):
    '''
    Returns
    Train, Validate, Test from SKLearn
    Sizes are 60% Train, 20% Validate, 20% Test
    '''
    train, test = train_test_split(df, test_size=.2, random_state=4343, stratify=stratify)

    train, validate = train_test_split(train, test_size=.2, random_state=4343, stratify=stratify)
    print(f'Dataframe: {df.shape}', '100%')
    print(f'Train: {train.shape}', '| ~60%')
    print(f'Validate: {validate.shape}', '| ~20%')
    print(f'Test: {test.shape}','| ~20%')

    return train, validate, test



def summarize(df):
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe. For
    each column, it returns the column name, the number of
    unique values in the column, the unique values themselves,
    the number of null values in the column, and the data type of the column.
    The resulting dataframe is sorted by the 'Number of Unique Values' column in ascending order.

    returns:
        pandas dataframe
    """
    data = []
    # Loop through each column in the dataframe
    for column in df.columns:
        # Append the column name, number of unique values, unique values, number of null values, and data type to the data list
        data.append(
            [
                column,
                df[column].nunique(),
                df[column].unique(),
                df[column].isna().sum(),
                df[column].dtype
            ]
        )

        check_columns = pd.DataFrame(
        data,
        columns=[
            "Column Name",
            "Number of Unique Values",
            "Unique Values",
            "Number of Null Values",
            "dtype"],
    ).sort_values(by="Number of Unique Values")
   
    # Create a pandas dataframe from the data list, with column names 'Column Name', 'Number of Unique Values', 'Unique Values', 'Number of Null Values', and 'dtype'
    # Sort the resulting dataframe by the 'Number of Unique Values' column in ascending order
    return check_columns



def organize_columns(train):
    '''
    Distinguishes between numeric and categorical data types
    Only selecting columns that would be relevant to visualize, no encoded data.
    '''
    cat_cols, num_cols = [], []
    explore = train
    for col in explore:
        # check to see if its an object type,
        # if so toss it in categorical
        if train[col].dtype == 'O':
            cat_cols.append(col)
        # otherwise if its numeric:
        else:
            # check to see if we have more than just a few values:
            # if thats the case, toss it in categorical
            if train[col].nunique() < 5:
                cat_cols.append(col)
            else:
                num_cols.append(col)
    return cat_cols, num_cols

def check_cat_distribution(df,target='tax_value'):
    '''
    Loop through a df and check their respective distributions.
    This is to be used with categorical datatypes, since the only 
    plot used is a countplot, with a target used as the hue to compare.
    '''
    
    for col in df:
        plt.figure(figsize=(12.5,8))
        sns.countplot(data=df,x=col,alpha=0.8,linewidth=.4,edgecolor='black')
        plt.title(col)
        plt.show()
        print('''-------------------------------------------------------------''')\
    

def check_num_distribution(df,dataset='train',target='tax_value'):
    '''
    Loop through a df and check their respective distributions.
    This is to be used with numerical datatypes, since the 
    plots used are hist plot and box plot, with a target used as the hue to compare.
    '''
    for col in df:
        sns.histplot(data=dataset, x=df[col],hue='tax_value')
        t = col.lower()
        plt.title(t)
        plt.show()
        sns.boxplot(data=dataset, x=col,hue='tax_value')
        plt.title(t)
        plt.show()
        print('''-------------------------------------------------------------''')

###               ###
# #     Scaler    # #
###               ###

def QuickScale(x_train, x_validate, x_test, linear=True, scaler='MinMax'):
    '''
    Produces data scaled with each respective style, will utilize all unless specificied otherwise.

    Arguments: x_train = desired data frame; and respected validate and test, Linear= True or False

    Returns: 6 or 2 arrays that would need to be assigned
    '''
    # Check for linear keyword argument to choose how to scale.
    if linear==True:
        mmscaler = MinMaxScaler()
        nscaler = StandardScaler()
        rscaler = RobustScaler()
    else:
        # Non Linear Scaler 
        qscaler = QuantileTransformer()
        # train
        x_train_scaled = qscaler.fit_transform(x_train)
        # validate
        x_val_scaled = qscaler.transform(x_validate)
        # test
        x_test_scaled = qscaler.transform(x_test)
        return x_train_scaled, x_val_scaled, x_test_scaled

        ##### change scaler to be kwarg to reduce output -- , scaler= #####
        # train
    if scaler == 'MinMax':
        x_train_scaled = mmscaler.fit_transform(x_train)
        x_val_scaled = mmscaler.transform(x_validate)
        x_test_scaled = mmscaler.transform(x_test)
        return x_train_scaled, x_val_scaled, x_test_scaled

    elif scaler == 'Standard':
        x_train_scaled = nscaler.fit_transform(x_train)
        x_val_scaled = nscaler.transform(x_validate)
        x_test_scaled = nscaler.transform(x_test)
        return x_train_scaled, x_val_scaled, x_test_scaled

    elif scaler == 'Robust':
        x_train_scaled = rscaler.fit_transform(x_train)
        x_val_scaled = rscaler.transform(x_validate)
        x_test_scaled = rscaler.transform(x_test)
        return x_train_scaled, x_val_scaled, x_test_scaled
    
    else:
        raise ValueError('Select a valid scaler.')