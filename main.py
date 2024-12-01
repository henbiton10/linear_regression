import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    house_prices = pd.read_csv(filepath_or_buffer='HousePricesHW1.csv')
    ax = plt.gca()
    plt.hist(house_prices['Price'], bins=30)
    max_price = house_prices.max()['Price']
    expensive = house_prices[house_prices['Price']>10000000]
    plt.title('Histogram')
    plt.ticklabel_format(axis='x',style='plain')
    plt.xlabel('price')
    plt.xlim(0, max_price)
    plt.xticks(np.arange(0, max_price + 1000000, 2000000))
    plt.show()