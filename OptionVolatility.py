# Import necessary libraries
import math
import matplotlib.pyplot as plt
import imageio
import numpy as np
import pandas as pd
import os
import calendar
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import date, datetime
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

# Uncomment line below to show plots in Jupiter notebook
# %matplotlib inline

# Section definition of functions **************************************************************************************

# Section calculation of implied volatility functions *****


# Black-Scholes call price
def call(s, k, t, r, sigma):
    d1 = math.log(s / (k * math.exp(-r * t))) / math.sqrt(math.pow(sigma, 2) * t) + \
         math.sqrt(math.pow(sigma, 2) * t) / 2
    d2 = d1 - math.sqrt(math.pow(sigma, 2) * t)
    c = s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
    return c


# Black-Scholes put price
def put(s, k, t, r, sigma):
    d1 = math.log(s / (k * math.exp(-r * t))) / math.sqrt(math.pow(sigma, 2) * t) + \
         math.sqrt(math.pow(sigma, 2) * t) / 2
    d2 = d1 - math.sqrt(math.pow(sigma, 2) * t)
    p = -s * norm.cdf(-d1) + k * math.exp(-r * t) * norm.cdf(-d2)
    return p


# Volatility Partial differential of Black-Scholes
def vega(s, k, t, r, sigma):
    d1 = math.log(s / (k * math.exp(-r * t))) / math.sqrt(math.pow(sigma, 2) * t) + \
         math.sqrt(math.pow(sigma, 2) * t) / 2
    vol = (1 / math.sqrt(2 * math.pi)) * s * math.sqrt(t) * math.exp(-(norm.cdf(d1) ** 2) * 0.5)
    return vol


# Use Newton's method to calculate the implied volatility of call option with price c
def calculate_iv_call(s, k, t, r, c):
    # If speed to high probability that newtons method jumps to negative IV and stays there. Good start value is 0.5
    speed = 0.5
    # You can set the error which IV is calculated for
    error = 0.001
    # Start values
    sigma_old = 0.1
    sigma_new = 0.2

    # While the difference between new and old sigma is bigger than error, newtons method of Black-Scholes with partial-
    # derivative vega is applied
    while abs(sigma_new - sigma_old) >= error:
        sigma_old = sigma_new
        sigma_new = sigma_old - ((call(s, k, t, r, sigma_new) - c) / vega(s, k, t, r, sigma_new)) * speed

        # In case sigma goes negative try again with different values and slower speed
        if sigma_new < -0.5:
            speed = 0.02
            sigma_old = 0.1
            sigma_new = 0.2
            while abs(sigma_new - sigma_old) >= error:
                sigma_old = sigma_new
                sigma_new = sigma_old - ((call(s, k, t, r, sigma_new) - c) / vega(s, k, t, r, sigma_new)) * speed
                if sigma_new < 0:
                    sigma_new = np.nan
                    break
            break
    return sigma_new


# Use Newton's method to calculate the implied volatility of put option with price c
def calculate_iv_put(s, k, t, r, c):
    # If speed to high, there is probability that newtons method jumps to negative IV and stays there.
    # Good start value is 0.5
    speed = 0.5
    # You can set the error which IV is calculated for
    error = 0.001
    # Start values
    sigma_old = 0.1
    sigma_new = 0.2

    # While the difference between new and old sigma is bigger than error, newtons method of Black-Scholes with partial-
    # derivative vega is applied
    while abs(sigma_new - sigma_old) >= error:
        sigma_old = sigma_new
        sigma_new = sigma_old - ((put(s, k, t, r, sigma_new) - c) / vega(s, k, t, r, sigma_new)) * speed

        # In case sigma goes negative try again with different values and slower speed
        if sigma_new < -0.5:
            speed = 0.02
            sigma_old = 0.1
            sigma_new = 0.2
            while abs(sigma_new - sigma_old) >= error:
                sigma_old = sigma_new
                sigma_new = sigma_old - ((put(s, k, t, r, sigma_new) - c) / vega(s, k, t, r, sigma_new)) * speed
                if sigma_new < 0:
                    sigma_new = np.nan
                    break
            break
    return sigma_new


# This function returns the time in years from today to third friday in month (the expire day of options on the Eurex
# exchange) of string time with format 'YYYYMM'
# This function is based on the ideas of this website:
# https://stackoverflow.com/questions/28680896/how-can-i-get-the-3rd-friday-of-a-month-in-python/28681204#28681204
def time_to_expiration(time):
    year = int(time[:4])
    month = int(time[4:])
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    month_expire = c.monthdatescalendar(year, month)
    # Eurex options expire on the third friday of the month
    third_friday = [day for week in month_expire for day in week if
                    day.weekday() == calendar.FRIDAY and
                    day.month == month][2]
    business_days = np.busday_count(date.today(), third_friday)
    # Half a day is added for the current day
    business_days += 0.5
    # Assumption that a year has 252 business days
    return business_days / 252


# Section of functions to manipulate df *****

# This function linearly interpolates NaN values in an array
# This function is based on the ideas of this website:
# https://stackoverflow.com/questions/36455083/working-with-nan-values-in-matplotlib/36457704
def interpolate_gaps(values):
    values = np.asarray(values)

    # Create list from array to count nan
    list_values = values.tolist()
    for i in range(len(values)):
        list_values[i] = str(list_values[i])
    # If the list contains less than 2 non nan values return entered list
    if list_values.count('nan') >= len(list_values)-1:
        return values
    else:
        # This code interpolates the values
        i = np.arange(values.size)
        valid = np.isfinite(values)
        filled = np.interp(i, i[valid], values[valid])

        # Now only interpolate values that lie within real values.
        # So if list starts with nan it stays nan until first value. Same at end.
        x = 0
        while filled[x] == filled[x+1] and x < len(filled)-2:
            filled[x] = np.nan
            x += 1
        x = -1
        while filled[x] == filled[x-1]:
            filled[x] = np.nan
            x -= 1
        return filled


# Interpolate all columns and rows in a dataframe
def interpolate_df(data):
    for column in data.columns:
        data[column] = interpolate_gaps(data[column])
    for row in data.index:
        data.loc[row] = interpolate_gaps(data.loc[row])
    return data


# Function to check if option has bid and ask price.
# Returns bid and ask dataset which only contains values when at same position the other dataset also has a value.
def check_bid_and_ask(b, a):
    for column in a.columns:
        for strike in a.index:
            if str(a.loc[strike, column]) == 'nan':
                b.loc[strike, column] = np.NaN
    for column in b.columns:
        for strike in b.index:
            if str(b.loc[strike, column]) == 'nan':
                a.loc[strike, column] = np.NaN
    return b, a


# Changes a df which contains option prices to the IV of these option prices
def change_prices_to_iv(prices, putt_call):
    share_p = share_price(share)
    interest = interest_rate()
    if putt_call == 'Call':
        for column in prices.columns:
            time_exp = time_to_expiration(column)
            # Get interest rate for this duration and change it to continuously compounding rate
            interest_exp = math.log(interest_days(time_exp * 252, interest)+1)
            for strike in prices.index:
                prices.loc[strike, column] = \
                    calculate_iv_call(share_p, strike, time_exp, interest_exp, prices.loc[strike, column])
    else:
        for column in prices.columns:
            time_exp = time_to_expiration(column)
            # Get interest rate for this duration and change it to continuously compounding rate
            interest_exp = math.log(interest_days(time_exp * 252, interest)+1)
            for strike in prices.index:
                prices.loc[strike, column] = \
                    calculate_iv_put(share_p, strike, time_exp, interest_exp, prices.loc[strike, column])
    return prices


# Assumption that true option price is in the middle of Bid and Ask.
# Input are 2 df with bid and with ask prices, returns dataframe with the mean of bid and ask
def price(b, a):
    p = pd.concat((b, a))
    p = p.groupby(p.index).mean()
    return p


# Delete all columns of df that only contain nan
def delete_nan_columns(df):
    for column in df.columns:
        nan_in_col = 0
        nb_rows = 0
        for row in df.index:
            if str(df.at[row, column]) == 'nan':
                nan_in_col += 1
            nb_rows += 1
        if nan_in_col == nb_rows:
            df = df.drop(column, 1)
    return df


# Section of functions to plot data *****

# Option to store the plot as png in a folder png in the current working directory
def store_plot(m):
    print('Do you want to save the plot as a png? Enter y for yes, n for no:')
    store = input()

    if store == 'y':
        current_path = os.getcwd()
        path = current_path + "/png"
        try:
            os.mkdir(path)
        except OSError:
            print('Creation of the directory %s failed. Maybe already exists.\n' % path)
        else:
            print('Successfully created the directory %s \n' % path)
        filename = path + '/' + share + call_putt + m + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plt.savefig(filename, dpi=300)


# Plot the iv of one maturity date
def plot_one_maturity(iv, mat):
    plt.plot(iv.index, iv[mat])
    plt.scatter(iv.index, iv[mat], s=10)
    plt.ylabel('Implied Volatility')
    plt.xlabel('Strike price')
    plt.title('IV ' + share + ' ' + call_putt + ' options for maturity ' + mat)
    plt.annotate(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), (0, 0), (10, -25), xycoords='axes fraction',
                 textcoords='offset points', va='top')
    store_plot(mat)
    plt.show()
    plt.close()


# Plot the iv for two maturity dates
def plot_two_maturities(iv, one, two):
    mat = one + '-' + two
    plt.plot(iv.index, iv[one], label='_nolegend_')
    plt.plot(iv.index, iv[two], label='_nolegend_')
    plt.scatter(iv.index, iv[one], s=10)
    plt.scatter(iv.index, iv[two], s=10)
    plt.ylabel('Implied Volatility')
    plt.xlabel('Strike price')
    plt.title('IV ' + share + ' ' + call_putt + ' options')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.annotate(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), (0, 0), (10, -25), xycoords='axes fraction',
                 textcoords='offset points', va='top')
    store_plot(mat)
    plt.show()
    plt.close()


# Plot the iv of all maturity dates
def plot_all(iv):
    mat = 'ALL'

    # Plot all columns of the df
    for column in iv.columns:
        plt.plot(iv.index, iv[column])
    plt.ylabel('Implied Volatility')
    plt.xlabel('Strike price')
    plt.title('IV ' + share + ' ' + call_putt + ' options')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.annotate(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), (0, 0), (0, -25), xycoords='axes fraction',
                 textcoords='offset points', va='top')
    store_plot(mat)
    plt.show()
    plt.close()


# create a gif of the volatility surface
# This function is based on the ideas of this website:
# https://python-graph-gallery.com/342-animation-on-3d-plot/
def volatility_surface_gif(df):

    # Change dataset to individual x,y,z points
    df = df.unstack().reset_index()
    df.columns = ["X", "Y", "Z"]
    df = df.dropna()

    # Change maturity date to time to maturity in years
    for strike in df.index:
        df.at[strike, 'X'] = time_to_expiration(str(df.at[strike, 'X']))
    x = df['X'].values
    y = df['Y'].values
    z = df['Z'].values
    images = []
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_path = os.getcwd()
    path = current_path + "/3D"
    try:
        os.mkdir(path)
    except OSError:
        print('Creation of the directory %s failed. Maybe already exists.\n' % path)
    else:
        print('Successfully created the directory %s \n' % path)
    print('The 3D animation is created now. This might take a while.')

    # Create a picture of the plot for every 2nd possible angle
    for angle in range(0, 360, 2):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # cmap is the color of the plot, different options available
        ax.plot_trisurf(list(x), list(y), list(z),
                        cmap='nipy_spectral', edgecolor='none', vmin=df['Z'].min(), vmax=df['Z'].max())
        ax.view_init(20, angle)
        plt.ylabel('Strike')
        plt.xlabel('Expiration in years')
        plt.title('Implied volatility surface ' + share + ' ' + call_putt + ' options')
        plt.annotate(time, (0, 0), (10, -10), xycoords='axes fraction',
                     textcoords='offset points', va='top')
        # Plot has to be saved as png to create giff from it. Change dpi to change quality
        filename = current_path + '/3D/' + 'IV' + str(angle) + '.png'
        plt.savefig(filename, dpi=100)
        plt.close()
        images.append(imageio.imread(filename))
        # Remove the png, not needed anymore
        os.remove(filename)
    # Duration influences the speed of the giff
    duration = 0.04
    # Create the giff
    imageio.mimwrite(current_path + '/3D/' + share + call_putt + '3DVolatilitySurface' +
                     datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.gif', images, duration=duration, loop=1)
    print('The 3D animation of the volatility surface is stored at: ' + current_path + '/3D/' + 'implied_volatility' +
          share + '.gif\n')


# Section of functions to download data from internet *****

# This method returns the share price of a ticker from cash.ch with a delay of about 15 minutes
def share_price(ticker):
    share_price_url = {'ABBN': 'abb-n-1222171', 'ADEN': 'adecco-group-n-1213860', 'CSGN': 'cs-group-n-1213853',
                       'GEBN': 'geberit-n-3017040', 'GIVN': 'givaudan-n-1064593', 'BAER': 'julius-baer-grp-n-10248496',
                       'LHN': 'lafargeholcim-n-1221405', 'LONN': 'lonza-grp-n-1384101', 'NESN': 'nestle-n-3886335',
                       'NOVN': 'novartis-n-1200526', 'CFR': 'ciefinrichemont-n-21048333', 'ROG': 'roche-hldg-g-1203204',
                       'SGSN': 'sgs-rg-249745', 'SIKA': 'sika-n-41879292', 'UHR': 'the-swatch-grp-i-1225515',
                       'SLHN': 'swiss-life-hldg-n-1485278', 'SREN': 'swiss-re-n-12688156', 'SCMN': 'swisscom-n-874251',
                       'UBSN': 'ubs-group-n-24476758', 'ZURN': 'zurich-insur-gr-n-1107539'}
    if ticker == 'l':
        share_l = list(share_price_url.keys())
        return share_l
    else:
        url = 'https://www.cash.ch/aktien/' + share_price_url[ticker] + '/swl/chf'
        html = urlopen(url)
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.findAll('li', {'class': 'first'})
        p_share = 0
        # The price of the share should be the only number in the row
        for row in rows:
            for cell in row.findAll(["span", "span"]):
                try:
                    p_share = float(cell.get_text().replace("'", ''))
                except ValueError:
                    continue
        if p_share == 0:
            print('There seems to be an Error with the share_price function. Check if the url is still valid and check '
                  'if on the website if the price is still store in li with class first.')
        return p_share


# Create two dataframes, one with all bid and one with all ask prices of options on share symbol from eurexchange.com
def eurex_prices(share_symbol):
    # First find all available maturity dates
    # Initial link, does not require maturity date
    share_url = {'ABBN': 'ABB-950336', 'ADEN': 'Adecco-951460', 'CSGN': 'Credit-Suisse-951996',
                 'GEBN': 'Geberit-945662',
                 'GIVN': 'Givaudan-952004', 'BAER': 'Julius-B-r-Gruppe-952060', 'LHN': 'LafargeHolcim-952052',
                 'LONN': 'Lonza-Group-945634', 'NESN': 'Nestl--945046', 'NOVN': 'Novartis-950294',
                 'CFR': 'Cie-Financi-re-Richemont-945618', 'ROG': 'Roche-Holding-951520', 'SGSN': 'SGS-952124',
                 'SIKA': 'Sika-952450', 'UHR': 'Swatch-Group-949466', 'SLHN': 'Swiss-Life-Holding-952194',
                 'SREN': 'Swiss-Re-950362', 'SCMN': 'Swisscom-947274', 'UBSN': 'UBS-951544',
                 'ZURN': 'Zurich-Insurance-Group-953098'}
    url = 'https://www.eurexchange.com/exchange-en/products/equ/opt/' + share_url[share_symbol]
    html = urlopen(url)
    soup = BeautifulSoup(html, "html.parser")
    maturity_dates = []
    row = soup.find('select', {'id': 'maturityDate'})
    for cell in row.findAll(["option", "th"]):
        maturity_dates.append(cell['value'])
    # Last element in list is not a maturity date
    maturity_dates = maturity_dates[:-1]

    # Now get all bid and ask prices for all maturity dates
    ask_all_dates = pd.DataFrame()
    bid_all_dates = pd.DataFrame()
    for mat_date in maturity_dates:
        share_url_maturity = {'ABBN': '47574', 'ADEN': '47628', 'CSGN': '49518',
                              'GEBN': '51300',
                              'GIVN': '51342', 'BAER': '48266',
                              'LHN': '51674',
                              'LONN': '52324', 'NESN': '53022', 'NOVN': '53336',
                              'CFR': '49260', 'ROG': '54166',
                              'SGSN': '54906',
                              'SIKA': '55054', 'UHR': '55974', 'SLHN': '55118',
                              'SREN': '55388', 'SCMN': '54732', 'UBSN': '55894',
                              'ZURN': '56534'}
        url = 'https://www.eurexchange.com/exchange-en/products/equ/opt/' + share_url_maturity[share_symbol] +\
              '!quotesSingleViewOption?callPut=' + call_putt + '&maturityDate=' + mat_date + '&changeDate=change+date'
        html = urlopen(url)
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", {"class": "dataTable"})
        rows = table.findAll("tr")
        eurex = pd.DataFrame()
        r = 0
        # Get values of all cells in table and safe them in dataframe
        for row in rows:
            c = 0
            for cell in row.findAll(["td", "th"]):
                # Change all missing values to correct NaN format
                if cell.get_text() == 'n.a.':
                    eurex.at[r, c] = np.nan
                else:
                    try:
                        eurex.at[r, c] = float(cell.get_text().replace(",", ''))
                    except ValueError:
                        eurex.at[r, c] = cell.get_text().replace(",", '')
                c += 1
            r += 1

        eurex_header = eurex.iloc[0]
        eurex = eurex[1:-2]
        eurex.columns = eurex_header

        eurex['Strike price'] = eurex['Strike price'].astype(float)
        eurex['Bid price'] = eurex['Bid price'].astype(float)
        eurex['Bid vol'] = eurex['Bid vol'].astype(float)
        eurex['Ask price'] = eurex['Ask price'].astype(float)
        eurex['Ask vol'] = eurex['Ask vol'].astype(float)
        eurex = eurex.set_index('Strike price')

        # From eurex df create bid and ask dataframes and add them to df with all maturity dates
        ask_date_header = [mat_date]
        ask_date = eurex.loc[:, ['Ask price']]
        ask_date.columns = ask_date_header
        ask_all_dates = ask_all_dates.append(ask_date, sort=True)
        bid_date_header = [mat_date]
        bid_date = eurex.loc[:, ['Bid price']]
        bid_date.columns = bid_date_header
        bid_all_dates = bid_all_dates.append(bid_date, sort=True)
    ask_all_dates = ask_all_dates.groupby(level=0).sum()
    ask_all_dates = ask_all_dates.replace(0, np.nan)
    bid_all_dates = bid_all_dates.groupby(level=0).sum()
    bid_all_dates = bid_all_dates.replace(0, np.nan)
    # Uncomment next two rows to store output of this function as csv
    # ask_all_dates.to_csv('ask.csv')
    # bid_all_dates.to_csv('bid.csv')
    return bid_all_dates, ask_all_dates


# Function to get forward interest rates for different durations and stores it in df
# Output is a pandas df with two columns. First column dtype object with text libor duration. Second column is dtype
# float64 with interest rates.
def interest_rate():
    url = 'https://www.finanzen.ch/zinsen'
    html = urlopen(url)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find('table', {'class': 'table tableAltColor table-small'})
    rows = table.findAll("tr")
    interest = pd.DataFrame()
    row_df = 0
    for row in rows:
        cell_df = 0
        for cell in row.findAll(["td", "th"]):
            interest.at[row_df, cell_df] = cell.get_text().replace(",", '')
            cell_df += 1
        row_df += 1
    interest = interest.iloc[1:, 0:2]
    interest.iloc[:, 1] = interest.iloc[:, 1].astype(float)
    return interest


# Find the relevant interest rate in interest_rate df for number of days in future
def interest_days(days, interest):
    if days >= 252:
        r = interest.at[7, 1]
    elif days >= 126:
        r = interest.at[6, 1]
    elif days >= 63:
        r = interest.at[5, 1]
    elif days >= 42:
        r = interest.at[4, 1]
    elif days >= 21:
        r = interest.at[3, 1]
    elif days >= 5:
        r = interest.at[2, 1]
    else:
        r = interest.at[1, 1]
    return r/100


# Section of the actual program ****************************************************************************************


# Program description
print('\nThis program analyses implied volatility (IV) of options on SMI stocks. The program only works properly during'
      ' market opening hours and runs with a delay of 15 minutes.')

# The program runs until it is stopped by input e
while True:

    # Loop to get a correct share symbol
    while True:
        # Enter lower or upper L because in Jupiter Notebook lower l looks like 1.
        print('Enter the symbol of the share to analyse. Enter l or L to show a list of all supported symbols:')
        share = input()
        share = share.upper()
        share_list = share_price('l')
        if share == 'L':
            print('The supported share symbols are:')
            print(share_list)
        else:
            if share in share_list:
                break
            else:
                print('This symbol is not supported\n')

    # Loop to select to analyse call or put
    while True:
        print('To analyse calls enter c, to analyse puts enter p:')
        call_putt = input()
        call_putt = call_putt.lower()
        if call_putt == 'p':
            call_putt = 'Put'
            break
        elif call_putt == 'c':
            call_putt = 'Call'
            break

    print('Downloading option prices. This might take a while.\n')

    # Download option prices and apply several functions to prepare data set
    bids, asks = eurex_prices(share)
    bids, asks = check_bid_and_ask(bids, asks)
    price_option = price(bids, asks)
    iv_df = change_prices_to_iv(price_option, call_putt)
    iv_df = interpolate_df(iv_df)
    iv_df = delete_nan_columns(iv_df)

    # The same stock can be analysed in different ways without downloading prices again
    while True:
        print('This program has 3 functions to analyse implied volatility: \n'
              '1. Plot IV for one stock for one maturity date.\n'
              '2. Plot IV for one stock for two maturity dates.\n'
              '3. Plot IV for one stock for all maturity dates and create 3D volatility surface.\n\n'
              'Enter 1, 2, 3, s to change stock, e to exit:')
        program_function = input()
        program_function = program_function.lower()

        if program_function == '1':
            while True:
                print('The possible maturity dates are:')
                print(list(iv_df))
                print('Please enter a maturity date:')
                maturity = input()
                if maturity in list(iv_df):
                    break
                print('There are no options for this maturity date.\n')
            plot_one_maturity(iv_df, maturity)

        if program_function == '2':
            print('The possible maturity dates are:')
            print(list(iv_df))
            while True:
                print('Please enter the first maturity date:')
                maturity_one = input()
                if maturity_one in list(iv_df):
                    break
                print('There are no options for this maturity date.\n')
            while True:
                print('Please enter the second maturity date:')
                maturity_two = input()
                if maturity_two in list(iv_df):
                    break
                print('There are no options for this maturity date.\n')
            plot_two_maturities(iv_df, maturity_one, maturity_two)

        if program_function == '3':
            plot_all(iv_df)
            volatility_surface_gif(iv_df)

        # Exit the loop of the current selected share
        if program_function == 's' or program_function == 'e':
            break

    # Exit the program completely
    if program_function == 'e':
        break


