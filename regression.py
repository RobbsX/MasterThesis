import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##### Define Functions #####

def getTotalUsers(shares):
    # Input: Shares as df of used shares

    total_users = pd.DataFrame(columns=['total'])

    for share_calc in shares:
        if share_calc == "AAL":
            break
        df = pd.read_csv("popularity_export/" + share_calc + ".csv")
        df.timestamp = pd.to_datetime(df.timestamp, format='%Y-%m-%d')  # converting to datetime %H:%M:%S
        df = df.groupby(pd.Grouper(key='timestamp', freq='1d'))['users_holding'].mean(0).round(0).reset_index()  # daily
        df.set_index('timestamp', inplace=True)  # replace index
        df.index = pd.to_datetime(df.index)  # Convert index to datetime type
        df.index = df.index.date

        if total_users.empty:
            total_users['total'] = df['users_holding']
        else:
            total_users['total'] += df['users_holding']
    return total_users


def getShareData(shareName, normalised=True, plot=False):
    # Get share data
    shareData = pd.read_csv("shareData.csv")
    shareData = shareData.loc[shareData['TICKER'] == shareName]
    # shareData.date = pd.to_datetime(shareData.date, format='%Y-%m-%d %H:%M:%S')  # converting to datetime
    shareData.index = shareData["date"]
    # Convert shareData to df and adapt index
    price = pd.DataFrame(shareData.PRC)
    price.index = pd.to_datetime(price.index)
    price.index = price.index.date

    # import one csv for popularity
    df = pd.read_csv("popularity_export/" + shareName + ".csv")
    df.timestamp = pd.to_datetime(df.timestamp, format='%Y-%m-%d')  # converting to datetime %H:%M:%S
    df = df.groupby(pd.Grouper(key='timestamp', freq='1d'))['users_holding'].mean(0).round(0).reset_index()  # daily
    df.set_index('timestamp', inplace=True)  # replace index
    df.index = pd.to_datetime(df.index)  # Convert index to datetime type
    df.index = df.index.date

    # Use normalized number of investors
    if normalised is True:
        total = getTotalUsers(shares)
        df['users_holding'] /= total['total']

    # JOIN popularity and price data on index. Weekends can be omitted, as no trade occurs there.
    merge = pd.merge(df, price, how='right', left_index=True, right_index=True)
    merge['Share'] = shareName

    # Visualise merge visually to check for outliers
    if shareName == "AAPL" and plot is True:  # shareName == 'AAPL' and False:
        fig, axs = plt.subplots(2)
        axs[0].plot(merge.index, merge['PRC'])  # price
        axs[1].plot(merge.index, merge['users_holding'])  # popularity
        # axs[2].scatter(merge['users_holding'], merge['PRC'])  # price/popularity
        # Make space for and rotate the x-axis tick labels
        fig.autofmt_xdate()
        # separate subplot titles
        axs[0].set_title('Price of AAPL')
        axs[1].set_title('Number of Investors of AAPL')
        fig.supxlabel('Date')
        axs[0].set_ylabel('Price [$]')
        axs[1].set_ylabel('No of Investors [%]')
        fig.show()
        fig.savefig('PNGs/AAPL_price_investors')

    # missing values are ok, as consistency of time series is important.
    # else, use "merge = merge.dropna()" to drop missing values or do prediction for NAs.
    return merge


def getShareQuintils(merge):
    # ToDo Normalise No of Investors. The app became more popular -> adapt!

    # Compute Quintils
    merge['priceChange'] = (merge['PRC'] / merge['PRC'].shift(1)) - 1  # daily PRC change
    merge['NIPctChange'] = (merge['users_holding'] / merge['users_holding'].shift(1)) - 1  # daily No Inv change
    # 3-day cumulative abnormal return (CAR)
    merge['CAR'] = (1 + merge['priceChange']).rolling(window=3).apply(np.prod, raw=True) - 1
    # Winsorize the CAR values at the 0.5 and 99.5 levels to remove extreme outliers
    merge['CAR'] = merge['CAR'].clip(lower=merge['CAR'].quantile(0.005), upper=merge['CAR'].quantile(0.995))
    # Calculate the three-day percent changes in the number of Robinhood investors
    merge['investorChange'] = merge['users_holding'].pct_change(3)
    # Create quintiles based on the CAR values
    merge['CARQuintile'] = pd.qcut(merge['CAR'], q=5, labels=False)
    # Group the data by the CAR quintiles and calculate the mean of the three-day percent changes in the NI for each quintile
    quintile_means = pd.DataFrame(merge.groupby('CARQuintile')['investorChange'].mean())
    # Add avgDailyReturn
    quintile_means['avgDailyReturn'] = merge.groupby('CARQuintile')['priceChange'].mean()

    # Analyse with Correlation. Positive -> momentum, Negative -> Mean-Reversion
    correlation = merge['investorChange'].corr(merge['CAR'])
    # Visualise Correlaction
    plot_corr = False
    if plot_corr is True:
        plt.scatter(merge['investorChange'], merge['CAR'])
        plt.xlabel('Investor Change')
        plt.ylabel('CAR')
        plt.title('Relationship between Investor Change and CAR')
        plt.show()
    return quintile_means, correlation


def getQuintileRegression(merge):
    # make regression analysis
    data = merge[['investorChange', 'CAR', 'CARQuintile']].dropna()
    data = pd.get_dummies(data, columns=['CARQuintile'], prefix='Q')
    data = sm.add_constant(data)
    # Fit multiple linear Regression Model
    model = sm.OLS(data['investorChange'], data[['const', 'CAR', 'Q_0.0', 'Q_1.0', 'Q_2.0', 'Q_3.0', 'Q_4.0']])
    regression_result = model.fit()
    # print(regression_result.summary())
    return regression_result


# Alternative way to get behaviour, straight-forward
def getStraightForward(merge):
    # Period length for calculating percent change
    n = 14
    # Calculate percent change in price over the last n days
    percent_change = merge['PRC'].pct_change(n)
    # Calculate change in the number of investors over the next n days
    investor_change = merge['users_holding'].pct_change(n).shift(-n)
    # Determine investment behavior
    investment_behavior = [
        'pos momentum' if pc > 0 and ic > 0 else 'neg momentum' if pc < 0 and ic < 0 else 'neg mean-reversion' if (
                pc > 0 and ic < 0) else 'pos mean-reversion' if (pc < 0 and ic > 0) else 'other' for pc, ic in
        zip(percent_change, investor_change)]

    # Create a temporary DataFrame to store the results for the current share
    temp_df = pd.DataFrame({
        'percent_change': percent_change,
        'investor_change': investor_change,
        'investment_behavior': investment_behavior
    })

    return temp_df


def getStraightForwardRegression(temp_df):
    # Make regression analysis
    data = temp_df[['investment_behavior', 'percent_change', 'investor_change']].dropna()
    data = pd.get_dummies(data, columns=['investment_behavior'], prefix='D')
    data = sm.add_constant(data)
    # Fit multiple linear regression model
    model = sm.OLS(data['investor_change'], data[
        ['const', 'percent_change', 'D_neg mean-reversion', 'D_pos mean-reversion', 'D_neg momentum',
         'D_pos momentum']])  # , 'D_other'
    regression_result = model.fit()
    # Print regression summary
    # print(regression_result.summary())
    return regression_result


def regression_to_latex(regression_result):
    # Get the coefficients, standard errors, and p-values
    coef = regression_result.params
    std_err = regression_result.bse
    p_values = regression_result.pvalues
    rsquared = "{:.4f}".format(regression_result.rsquared)
    observations = regression_result.model.data.orig_endog.array.size

    # Create the LaTeX table header
    latex_table = "\\begin{table}\n\\centering\n"
    latex_table += "\\caption{Regression Results}\n"
    latex_table += "\\begin{tabular}{lcc}\n"
    latex_table += "\\hline\n"
    latex_table += "Variable & Coefficient & (Standard Error) \\\\\n"
    latex_table += "\\hline\n"

    # Loop through the coefficients and add rows to the table
    for var, c, se, p in zip(coef.index, coef, std_err, p_values):
        significance = ''
        if p < 0.01:
            significance = '***'
        elif p < 0.05:
            significance = '**'
        elif p < 0.1:
            significance = '*'
        row = f"{var} & {c:.3f} & ({se:.3f}) {significance} \\\\\n"
        row = row.replace('*', '$*$')  # Replace asterisks with math mode delimiters
        latex_table += row

    # Add the table footer
    latex_table += "\\hline\n"
    latex_table += "R-squared & " + str(rsquared) + " \\\\\n"
    latex_table += "Number of Observations & " + str(observations) + " \\\\\n"  # ToDo Test no obsv!
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}"

    return latex_table



#### END Define Functions ####


# Results DFs
results_quintils = pd.DataFrame(columns=['Share', 'CARQuintile', 'investorChange', 'avgDailyReturn'])
result_straightforward = pd.DataFrame(columns=['Share', 'percent_change', 'investor_change', 'investment_behavior'])
regression_quintile_results = pd.DataFrame(
    columns=['Share', 'R-squared', 'Intercept', 'CAR', 'Q_0', 'Q_1', 'Q_2', 'Q_3', 'Q_4'])
regression_sf_results = pd.DataFrame(
    columns=['Share', 'R-squared', 'Intercept', 'pos mean-reversion', 'neg mean-reversion', 'pos momentum',
             'neg momentum'])
quintile_total = pd.DataFrame(columns=['Share', 'users_holding', 'PRC', 'priceChange', 'NIPctChange', 'CAR', 'investorChange', 'CARQuintile'])

# import saved csv of top 1% popular shares
dfCheck = pd.read_csv('usedStocks.csv')  # extracted from CheckStocks.py
dfCheck["TICKER"] = dfCheck["fname"].apply(lambda x: x.split('/')[1].split('.')[0])
# describe popularity of stocks (use breakpoint)
describe = dfCheck.maxPopul.describe()

# get shares for Loop
shares = dfCheck["TICKER"]

for share in shares:
    if share == "AAL":
        break

    # Done ToDo get alternative definitions of mean-reversion mom -> n% change
    # Get quintile
    merge = getShareData(share, normalised=True, plot=False)
    quintile_means, correlation = getShareQuintils(merge)
    quintile_means["Share"] = share
    quintile_means["CARQuintile"] = quintile_means.index
    results_quintils = pd.concat([results_quintils, quintile_means], ignore_index=True)

    # get merge_total / quintile_total
    quintile_total = quintile_total.append(merge)

    # Alternative StraightForward
    temp_df = getStraightForward(merge)
    temp_df["Share"] = share
    result_straightforward = result_straightforward.append(temp_df)

    # Get all regression results
    regression_quintile = getQuintileRegression(merge)
    regression_sf = getStraightForwardRegression(temp_df)

    if share == "AAPL":
        latexRegression = regression_to_latex(regression_quintile)
        print(latexRegression)

    # Store the regression results in the DataFrame
    regression_quintile_results = regression_quintile_results.append({
        'Share': share,
        'R-squared': regression_quintile.rsquared,
        'Intercept': regression_quintile.params['const'],
        'CAR': regression_quintile.params['CAR'],
        'Q_0': regression_quintile.params['Q_0.0'],
        'Q_1': regression_quintile.params['Q_1.0'],
        'Q_2': regression_quintile.params['Q_2.0'],
        'Q_3': regression_quintile.params['Q_3.0'],
        'Q_4': regression_quintile.params['Q_4.0']
    }, ignore_index=True)

    regression_sf_results = regression_sf_results.append({
        'Share': share,
        'R-squared': regression_sf.rsquared,
        'Intercept': regression_sf.params['const'],
        'pos mean-reversion': regression_sf.params['D_pos mean-reversion'],
        'neg mean-reversion': regression_sf.params['D_neg mean-reversion'],
        'pos momentum': regression_sf.params['D_pos momentum'],
        'neg momentum': regression_sf.params['D_neg momentum']
        # 'other': regression_sf.params['D_other']
    }, ignore_index=True)


# ------------ Analyse all shares -----------
result_quintile_avg = results_quintils.groupby(['CARQuintile']).mean().reset_index()
# ToDo Add num of stocks where mean-reversion & momentum

# Group by behaviour of result_straightforward
# plot_AAPL = True
# if plot_AAPL is True:
#    result_straightforward = result_straightforward[result_straightforward["Share"]=="AAPL"]
result_straightforward_avg = result_straightforward.groupby(['investment_behavior']).mean().reset_index()
count = result_straightforward.groupby(['investment_behavior']).size().reset_index(name='count')
result_straightforward_avg = result_straightforward_avg.merge(count, on="investment_behavior")



# Done ToDo for each Quintile in quintile_result_avg get "pos momentum", "neg momentum", "mean-reversion"
# Create an empty list to store the behaviour categories
behaviour_categories = []
# Iterate over the quintile_result_avg DataFrame
for index, row in results_quintils.iterrows():
    avg_daily_return = row['avgDailyReturn']
    investor_change = row['investorChange']

    # Categorize based on conditions
    if avg_daily_return > 0 and investor_change > 0:
        category = "pos momentum"
    elif avg_daily_return < 0 and investor_change < 0:
        category = "neg momentum"
    elif (avg_daily_return > 0 and investor_change < 0):
        category = "neg mean-reversion"
    elif (avg_daily_return < 0 and investor_change > 0):
        category = "pos mean-reversion"  # investors belief in raising prices
    else:
        category = "other"

    # Append the category to the behaviour_categories list
    behaviour_categories.append(category)
# Add the behaviour_categories list as a new column in the quintile_result_avg DataFrame
results_quintils['Behaviour'] = behaviour_categories

# get PNG of quintils of AAPL
PNGdf = results_quintils[results_quintils["Share"] == "AAPL"]
PNGdf = PNGdf.loc[:, PNGdf.columns != 'Share']
PNGdf["investorChange"] = PNGdf["investorChange"].apply(lambda x: x * 100)
PNGdf["avgDailyReturn"] = PNGdf["avgDailyReturn"].apply(lambda x: x * 100)
print("\n----------- Tbl of Quintile Avg -----------")
# print(PNGdf.to_latex(index=False, float_format="{:.2f}".format))

# get tbl of result_straightforward_avg
result_straightforward_avg["percent_change"] = result_straightforward_avg["percent_change"].apply(lambda x: x * 100)
result_straightforward_avg["investor_change"] = result_straightforward_avg["investor_change"].apply(lambda x: x * 100)
print("\n----------- Tbl of Extrapolate Avg -----------")
print(result_straightforward_avg.to_latex(index=False, float_format="{:.2f}".format))


# -------- Do total regressions -------

# quintile results all together
total_regression_quintile = getQuintileRegression(quintile_total)
print("\n----------- Tbl of Total Regression Quintile -----------")
print(regression_to_latex(total_regression_quintile))

# extrapolate results all together
total_regression_extrapolate = getStraightForwardRegression(result_straightforward)
print("\n----------- Tbl of Total Regression Extrapolate -----------")
print(regression_to_latex(total_regression_extrapolate))
