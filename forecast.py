import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric

if __name__ == '__main__':
    df = pd.read_csv('workorder.csv')
    df = df[['Date', 'CO']].dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    daily_df = df.resample('D').mean()
    d_df = daily_df.reset_index().dropna()

    d_df.sort_values(by=['Date'])
    sns.set_style('whitegrid')
    plt.figure(figsize = (25, 10))
    ax = plt.axes()

    sns.lineplot(x='Date', y='CO', data=d_df, color='#76b900')
    ax.xaxis.set_major_locator(plt.MaxNLocator('auto'))
    plt.title('WorkOrder Counts', fontsize=16)
    plt.xlabel('date', fontsize=16)
    plt.ylabel('workorder counts', fontsize=16)

    d_df.columns = ['ds', 'y']
    m = Prophet()
    m.fit(d_df)

    future = m.make_future_dataframe(periods=90)
    print(future.tail())
    forecast = m.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)

    df_cv = cross_validation(m, horizon='90 days')
    print(df_cv.head())
    df_p = performance_metrics(df_cv)
    print(df_p.head(5))

    fig3 = plot_cross_validation_metric(df_cv, metric='mape')

    plt.show()