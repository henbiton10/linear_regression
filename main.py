import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
import statsmodels.api as sm


def q2_data(house_prices):
    #create histogram for prices
    plt.hist(house_prices['Price'], bins=30)

    #get mean, median and std dev
    avg = float(house_prices['Price'].mean())
    median = float(house_prices['Price'].median())
    std_dev = float(house_prices['Price'].std())

    print({'mean': avg, 'median': median, 'standard deviation': std_dev})

    #check why standard deviation is so high
    max_price = float(house_prices.max()['Price']) #what is the most expensive house
    expensive = len(house_prices[house_prices['Price'] > 10000000]) #how many houses are more expensive then 10 million
    print({'expensive': expensive, 'max_price': max_price})

    #set histogram params
    plt.title('Histogram')
    plt.ticklabel_format(axis='x', style='plain')
    plt.xlabel('price')
    plt.xlim(0, max_price)

    #set jumps in x axis
    plt.xticks(np.arange(0, max_price + 1000000, 2000000))
    plt.show()

def create_generic_linear_regression(df: DataFrame, target: str, features: list[str]):
    column_names = df.columns.tolist() #get list of column names
    #validate data
    if not all(name in column_names for name in [*features, target]):
        raise ValueError('not all parameters passed are in the dataframe')
    reg = sm.formula.ols(formula=f'{target} ~ {' + '.join(features)}', data=df).fit()
    print(reg.summary())
    p_values = reg.pvalues #get pvalues
    r_squared = reg.rsquared # get r squared
    significant_vars = p_values[p_values < 0.05].index #get significant p_values
    print(f'r squared value is: {r_squared * 100}%') # convert r squared to precentage
    print('significant variables are: ', significant_vars)
    f_pvalue = reg.f_pvalue
    #check if model is statistically significant
    if f_pvalue < 0.05:
        print('the model is statistically significant')
    else:
        print('the model is NOT statistically significant')
    weights = reg.params #get the weight of each feature
    print('parameter weights: ', weights)


def q3_data(house_prices: DataFrame):
    features = [name for name in house_prices.columns.tolist() if name != 'Price'] #get all columns for first model
    #run m1
    create_generic_linear_regression(df=house_prices, target='Price', features=features)
    #check correlation between stors and dog parks for q3
    print('correlation between stores and dog parks is: ',house_prices['NumStores'].corr(house_prices['DogParkInd']))
    print('---------------------------------------------------------------------------------------------------------'
          '-----------------------------------------------------------------------------')
    #run m2
    create_generic_linear_regression(df=house_prices, target='Price', features=['MtrsToBeach','SqMtrs', 'Age']) #2nd models params


if __name__ == '__main__':
    #read the file
    data = pd.read_csv(filepath_or_buffer='HousePricesHW1.csv')
    if data is not None:
        q2_data(data) #run q2 commands
        q3_data(data) #run q3 commands

