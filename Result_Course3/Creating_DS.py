# Импортируем необходимые библиотеки:
import pandas as pd
import grequests

from bs4 import BeautifulSoup as BS

# Create global value for future manipulation
counter = 1

# Function for connection with web-site and restore data about price and cuisine from it
def url_connection(link):
    global counter  # Initialisation of global value (counter)

    url = 'https://www.tripadvisor.com' + link  # Create a url link
    print(counter, url)

    # Try to connect with web-site
    with grequests.Session() as session:
        # Initialisation of header to connect url
        session.headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' \
                                        '(KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36'

        # Try to connect with web site. If connection fall, or can't create values, so connection lost and
        try:
            resp = session.get(url, timeout=5)  # Connect with url with 5 second timeout and create a copy for next pasrser
            soup = BS(resp.content, 'html.parser')  # Use method BeautifulSoup
            price = soup.find(class_='_2mn01bsa').contents[0]  # Find information about price
            cuisine = soup.find('div', {'class': ['_60ofm15k', '_1XLfiSsv']}).contents  #  Find information about cuisine
        except:
            price = None
            cuisine = None
            print('no connection/empty data')

    counter += 1  # arise counter

    price = price if price and ('$' in price) else None  # Test, that information have data about price

    print(f'The costs is: {price}. The cuisine of restaurant is {cuisine}')

    return [price, cuisine]

pd.set_option('display.max_rows', 50)  # Show more rows
pd.set_option('display.max_columns', 50)  # Show more columns

# Read the file with data set
meals_ds = pd.read_csv('./main_task_new.csv')

# Create Data frame for the next work. Use only data with empty values in column Price Range and Cuisine Style
lost_ds = pd.DataFrame()
lost_ds['Restaurant_id'] = meals_ds[(meals_ds['Price Range'].isna()) |
                                    (meals_ds['Cuisine Style'].isna())]['Restaurant_id'].copy()
lost_ds['index_df'] = meals_ds[(meals_ds['Price Range'].isna()) | (meals_ds['Cuisine Style'].isna())].copy().index

for col in meals_ds.columns:
    meals_ds[col] = meals_ds[col].apply(lambda x: None if pd.isnull(x) else x)

try:
    # Try to open file to continue writing data (if file exists)
    restore_data = pd.read_csv('./frames.csv', sep='\t')
    f = open('frames.csv', 'a', encoding='utf-8')

    # Remove all data analyzed in previous iteration
    lost_ds['for_remove'] = [1 if i in restore_data['index_df'].values else 0 for i in lost_ds['index_df'].values]
    lost_ds = lost_ds.drop(lost_ds[lost_ds['for_remove'] == 1].index)

    print(len(lost_ds), '\n', lost_ds.head()) # Print data for analyzing the algorithm work

except:
    # If file does not exist, create one
    f = open('frames.csv', 'w', encoding='utf-8')
    f.write('index_df\tRestaurant_id\tPrice Range\tCuisine Style\n')
    print('File does not exist')

print('Number of url connection is: ', len(lost_ds))

# Cycle for writing data into file
for index in lost_ds.index:
    data_frame = url_connection(meals_ds.iloc[index]['URL_TA'])
    f.write(str(index) + '\t' + str(meals_ds.iloc[index]['Restaurant_id']) + '\t' + str(data_frame[0]) + '\t'
            + str(data_frame[1]) + '\n')

    # Every 100 iterations file open and close for exception to lost data
    if counter % 100 == 0:
        f.close()
        f = open('frames.csv', 'a', encoding='utf-8')

    # Every 1000 iterations make a decision, continue or break the cycle
    if counter % 1000 == 0:
        continue_ = input('Do you want to continue (y/n): ')
        if continue_ in 'Nn':
            break

f.close() # close the file
