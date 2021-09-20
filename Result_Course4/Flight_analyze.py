import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Module for plotting
import seaborn as sns # Module for plotting
from datetime import datetime  # Module for working with dates and time
from IPython.display import display

# # зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:
# !pip freeze > requirements.txt

pd.set_option('display.max_rows', 250)  # Show more rows
pd.set_option('display.max_columns', 250)  # Show more columns

# Dictionary of fuel consumption and fuel cost percent (of full cost) by month
gsm_cost = {
    1: [41435, 0.2175],
    2: [39553, 0.2],
    12: [47101, 0.21]
}

# List of columns with data
date_columns = ['scheduled_departure', 'scheduled_arrival', 'actual_departure', 'actual_arrival']

# Function for calculating percentiles
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)

    percentile_.__name__ = 'Q_%s' % (n // 25)
    return percentile_

# Creating class for computing statistics of flights
class flight_stat():

    global gsm_cost, date_columns

    # Create initiating parts
    def __init__(self):
        self.frame = pd.read_csv('./flights_data_set.csv') # read the file with dataset

        # Reform columns with data into datetime format
        for column in date_columns:
            self.frame[column] = self.frame[column].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))

        # Creating a set with aircrafts and it properties
        self.aircrafts = pd.read_csv('./aircrafts.csv', sep=';')

        # Creating the additional columns
        self.frame['actual_flight_time'] = self.frame['actual_arrival'] - self.frame['actual_departure']
        self.frame['actual_flight_time'] = self.frame['actual_flight_time'].apply(lambda x: x.total_seconds()/60)
        self.frame['scheduled_flight_time'] = self.frame['scheduled_arrival'] - self.frame['scheduled_departure']
        self.frame['scheduled_flight_time'] = self.frame['scheduled_flight_time'].apply(lambda x: x.total_seconds() / 60)
        self.frame['departure_delay'] = self.frame['scheduled_departure'] - self.frame['actual_departure']
        self.frame['departure_delay'] = self.frame['departure_delay'].apply(lambda x: x.total_seconds() / 60)
        self.frame['arrival_delay'] = self.frame['scheduled_arrival'] - self.frame['actual_arrival']
        self.frame['arrival_delay'] = self.frame['arrival_delay'].apply(lambda x: x.total_seconds() / 60)
        self.frame['schedule_flight_cost'] = self.frame.apply(lambda row: self.full_cost(row, 'schedule'), axis=1)
        self.frame['actual_flight_cost'] = self.frame.apply(lambda row: self.full_cost(row, 'actual'), axis=1)
        self.frame['flight_profit'] = self.frame['actual_amount_sum'] - self.frame['actual_flight_cost']
        self.frame['month'] = self.frame['actual_departure'].apply(lambda x: x.month)
        self.frame['flight_weekday'] = self.frame['actual_departure'].apply(lambda x: x.strftime('%A'))

        # Initiating the dataframe with race statistics
        self.races = pd.DataFrame()
        self.races['races'] = self.frame['flight_no'].value_counts().values
        self.races.index = self.frame['flight_no'].value_counts().index
        for race in self.races.index:
            self.races.at[race, 'model'] = self.frame[self.frame['flight_no'] == race]['model'].values[0]

    # Function for computing full coast of race
    def full_cost(self, row, type_):

        # Preparing branch that will be used in function
        if type_ == 'schedule':
            control_columns = ['scheduled_departure', 'scheduled_arrival', 'scheduled_flight_time']
        else:
            control_columns = ['actual_departure', 'actual_arrival', 'actual_flight_time']
        # Fuel cost for flight with tax
        gsm_flight_cost = gsm_cost[row[control_columns[0]].month][0] * 1.18
        # Fuel consumption in tons
        gsm_consumption = self.aircrafts[self.aircrafts['model'] == row['model']]['fuel_consumption'] / 1000
        # Computing flight time in hours
        flight_time = row[control_columns[2]] / 60

        # Computing total flight coast in next formula: tc=fuel_cost/(percent_of_tc) where:
        # fuel_cost = flight time (hours) * fuel consumption (tons per hours) * fuel_cost (rub per tons)
        # percent_of_tc = fuel percent of summary cost of race (by ria.ru data)
        total_cost = round((gsm_flight_cost * gsm_consumption * flight_time).values[0] /
                            gsm_cost[row[control_columns[0]].month][1], 2)

        return total_cost

    # Object for flight profit boxplot plotting
    def showing_flights(self):

        # Generate plotting area and axis parameters
        fig, axes = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [12, 1]})
        axes[0].set_title(f'Boxplot of profit by race')
        axes[0].xaxis.set_label_position('top')
        axes[0].xaxis.tick_top()
        axes[1].axis('tight')
        axes[1].axis('off')

        # Generate the statistics parameters table ana
        column_stats = self.frame.groupby(by='flight_no')['flight_profit'].agg(['count', 'mean', 'std', 'median',
                                                                                'min', 'max', percentile(25),
                                                                                percentile(75)])

        # Create table object in plotting area
        table = axes[1].table(cellText=np.round(column_stats.values.tolist(), 1),
                              colLabels=column_stats.columns,
                              rowLabels=column_stats.index)
        # Configure table parameters
        table.auto_set_font_size(False)
        table.set_fontsize(11)

        # Configure boxplot object and it parameters
        sns.boxplot(ax=axes[0],
                    x='flight_no',
                    y='flight_profit',
                    data=self.frame,
                    )
        # Show the plotting area
        plt.show()

    # Object for generating races statistics table
    def stat(self):

        # Common statistics of races profit
        display(self.frame.groupby('flight_no')['flight_profit'].describe())

        # Statistics of weekday divided by month
        months_ = self.frame['month'].unique()
        week_days = self.frame['flight_weekday'].unique()

        # Generate race statistics by month and weekday
        for month in months_:

            self.races.loc[:, str(month) + '_costs'] = 0  # race cost
            self.races.loc[:, str(month) + '_mean_dep_delay'] = 0  # mean departure delay
            self.races.loc[:, str(month) + '_max_dep_delay'] = 0  # max departure delay
            self.races.loc[:, str(month) + '_max_dep_delay_day'] = 0  # max departure delay day
            self.races.loc[:, str(month) + '_mean_ft_dif'] = 0  # mean flight time difference
            self.races.loc[:, str(month) + '_max_ft_dif'] = 0  # max flight time difference
            self.races.loc[:, str(month) + '_max_ft_day'] = 0  # max flight time day

            for day in week_days:

                self.races.loc[:, str(month) + '_' + day + '_profit'] = None  # Mean profit combine by month and day
                self.races.loc[:, str(month) + '_' + day + 'bc_percent'] = None  # Mean percent of business class boarding
                self.races.loc[:, str(month) + '_' + day + 'ec_percent'] = None  # Mean percent of economy class boarding
                for race in self.races.index:

                    indexes = self.frame[(self.frame['month'] == month)
                                         & (self.frame['flight_weekday'] == day)
                                         & (self.frame['flight_no'] == race)].index
                    try:
                        self.races.at[race, str(month) + '_' + day + '_profit'] = \
                            self.frame.loc[indexes, 'flight_profit'].mean()
                        race_bc = self.frame.loc[indexes, 'buisness_ticket_count'].mean()
                        race_ec = self.frame.loc[indexes, 'economy_ticket_count'].mean()
                        ac_ind = self.aircrafts[self.aircrafts['model'] == self.races.loc[race, 'model']].index
                        ac_bc = self.aircrafts.loc[ac_ind, 'business_ticket_count'].values[0]
                        ac_ec = self.aircrafts.loc[ac_ind, 'economy_ticket_count'].values[0]

                        self.races.at[race, str(month) + '_' + day + '_bc_percent'] = (race_bc / ac_bc) * 100
                        self.races.at[race, str(month) + '_' + day + '_ec_percent'] = (race_ec / ac_ec) * 100

                    except ValueError:
                        self.races.at[race, str(month) + '_' + day + '_profit'] = None

            for race in self.races.index:

                indexes = self.frame[(self.frame['month'] == month)
                                     & (self.frame['flight_no'] == race)].index
                self.races.at[race, str(month) + '_costs'] = self.frame.loc[indexes, 'actual_flight_cost'].mean()
                self.races.loc[race, str(month) + '_mean_dep_delay'] = self.frame.loc[indexes, 'departure_delay'].mean()
                self.races.loc[race, str(month) + '_max_dep_delay'] = self.frame.loc[indexes, 'departure_delay'].min()
                max_index = self.frame.loc[indexes, 'departure_delay'].min()
                max_index = self.frame.loc[indexes][self.frame.loc[indexes, 'departure_delay'] == max_index].index
                self.races.loc[race, str(month) + '_max_dep_delay_day'] = \
                                                            self.frame.loc[max_index, 'scheduled_departure'].values[0]
                self.races.loc[race, str(month) + '_mean_ft_dif'] = self.frame.loc[indexes, 'actual_flight_time'].mean()
                self.races.loc[race, str(month) + '_max_ft_dif'] = self.frame.loc[indexes, 'actual_flight_time'].max()
                max_dif = self.frame.loc[indexes, 'actual_flight_time'].max()
                max_dif_index = self.frame.loc[indexes][self.frame.loc[indexes, 'actual_flight_time'] == max_dif].index

                self.races.loc[race, str(month) + '_max_ft_day'] = \
                                                        self.frame.loc[max_dif_index, 'scheduled_departure'].values[0]

        display(self.races)

    # Object for writing data tables into files
    def write_file(self):
        self.frame.to_excel('result_table.xlsx')
        self.races.to_excel('races_table.xlsx')


flight_set = flight_stat()
flight_set.stat()
flight_set.showing_flights()
flight_set.write_file()
