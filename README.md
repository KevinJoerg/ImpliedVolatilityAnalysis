# Implied Volatility Analysis Project

The purpose of this program is to display implied volatility (IV) of options on stocks that are in the SMI. It consists of one single Python file. It currently runs with a delay of 15 minutes.

##Prerequisites
The program makes use of the following python libraries that need to be installed:
- os
- math
- numpy
- pandas
- matplotlib.pyplot
- imageio
- calendar
- urllib.request
- bs4
- datetime
- scipy.stats
- mpl_toolkits.mplot3d

## Running the program
The python file can be run in a python development environment. It is tested in PyCharm and Jupiter Notebook. To show the plots in Jupiter Notebook uncomment the line %matplotlib inline.

The program has two navigation levels. In the first one a stock can be selected by entering its symbol and you can select to analyse call or put options. In the second level you can choose 3 different functions to analyse the implied volatility:
1. Plot the "volatility smile" for one maturity date
2. Compare the "volatility smile" for two different maturity dates
3. Plot the IV for all maturity dates and create a 3D animation of the volatility surface

## About the program

The program makes use of three different kind of functions:
1. Functions that gather data from the internet through web scraping
2. Functions to manipulate the datasets and calculate IV
3. Functions to display the implied volatility graphically 

### Attention:
As the functions that gather data from the internet are dependent on the websites they take the data from, the program might suddenly stop working. The program last tested positively to run on 5. April 2019. 
When the websites change these dependent functions have to be adapted to the new websites. Either just the link changes, then the url dictionaries can simply be updated. When the whole website changes, maybe more has to be updated. 
