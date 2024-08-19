import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx


def create_highest_lowest_pm25_all(directory):
    max_aqi_value = float('-inf')
    min_aqi_value = float('inf')
    location_max_aqi = None
    location_min_aqi = None

    # Iterate through each CSV file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Extract location name from the filename
            location = filename.split('.')[0]
            
            # Read the CSV file into a DataFrame
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            
            # Find the maximum and minimum PM2.5 AQI values in the current dataframe
            max_value = df['PM2.5 AQI'].max()
            min_value = df['PM2.5 AQI'].min()

            # Update the overall max and min if the current location's values are higher/lower
            if max_value > max_aqi_value:
                max_aqi_value = max_value
                location_max_aqi = location

            if min_value < min_aqi_value:
                min_aqi_value = min_value
                location_min_aqi = location

    return max_aqi_value, min_aqi_value, location_max_aqi, location_min_aqi

def create_highest_lowest_pm25_one_location(directory, location_page):
    # Initialize max and min AQI values
    max_aqi_value = float('-inf')
    min_aqi_value = float('inf')

    # Construct the filename based on the location_page variable
    filename = f"{location_page}.csv"

    # Check if the file exists in the directory
    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Find the maximum and minimum PM2.5 AQI values in the dataframe
        max_aqi_value = df['PM2.5 AQI'].max()
        min_aqi_value = df['PM2.5 AQI'].min()
    else:
        raise FileNotFoundError(f"No file found for location: {location_page}")

    # Return the max and min AQI values along with the location name
    return max_aqi_value, min_aqi_value, location_page, location_page

def create_top5_highest_pm25_all(directory):
    # Dictionary to store the average PM2.5 AQI for each location
    avg_pm25_aqi = {}
    
    # Iterate through each CSV file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Extract location name from the filename
            location = filename.split('.')[0]
            
            # Read the CSV file into a DataFrame
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            
            # Calculate the average PM2.5 AQI for the current location
            avg_pm25_aqi[location] = df['PM2.5 AQI'].mean()

    # Convert the dictionary to a DataFrame for easier manipulation
    avg_pm25_aqi_df = pd.DataFrame(list(avg_pm25_aqi.items()), columns=['Location', 'Average PM2.5 AQI'])

    # Sort the DataFrame by average PM2.5 AQI in descending order
    avg_pm25_aqi_df = avg_pm25_aqi_df.sort_values(by='Average PM2.5 AQI', ascending=False).head(5)

    # Sort the top 5 locations in ascending order
    avg_pm25_aqi_df = avg_pm25_aqi_df.sort_values(by='Average PM2.5 AQI', ascending=True)

    return avg_pm25_aqi_df

def create_daily_dataframe_dict(directory):
    daily_dataframes = {}

    # Iterate through each CSV file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Extract location name from the filename
            location = filename.split('.')[0]
            
            # Read the CSV file into a DataFrame
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            
            # Convert the 'date' column to a datetime object
            df['date'] = pd.to_datetime(df['date'])
            
            # Store the DataFrame in the dictionary with the location as the key
            daily_dataframes[location] = df

    return daily_dataframes

def plot_air_quality_status(location_page):
    # Construct the file path based on the location
    file_path = f'status_dataframes/{location_page}.csv'
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Ensure the status_columns order is consistent with your preference
    status_columns = ['Good', 'Hazardous', 'Moderate', 'Unhealthy', 'Unhealthy for Sensitive Groups', 'Very Unhealthy']
    
    # Pivot the data to match the desired status order
    df_pivot = df.pivot_table(values='Count', index='Location', columns='Air Quality', aggfunc='sum').reindex(columns=status_columns).fillna(0)
    
    # Extract values for the selected location
    status_values = df_pivot.loc[location_page, status_columns]
    
    # Sort the status values by the highest count
    status_values_sorted = status_values.sort_values(ascending=True)
    
    # Determine the color: blue for the max value, gray for others
    colors = ['blue' if v == status_values_sorted.max() else 'gray' for v in status_values_sorted]
    
    # Create a vertical bar chart
    fig = plt.figure(figsize=(10, 6))
    plt.barh(status_values_sorted.index, status_values_sorted.values, color=colors)
    
    # Add a title and labels
    plt.title(f'PM 2.5 Air Quality Status in {location_page}', fontsize=16)
    
    # Show the plot
    st.pyplot(fig)

location_page = 'Beijing'
homepage = True


# Call the function with the directory containing the CSV files
directory = 'daily_dataframes' 
max_aqi_value, min_aqi_value, location_max_aqi, location_min_aqi = create_highest_lowest_pm25_all(directory)

directory = 'daily_dataframes_without_outlier'
avg_pm25_aqi_df = create_top5_highest_pm25_all(directory)

directory = 'daily_dataframes'
daily_dataframes = create_daily_dataframe_dict(directory)

with st.sidebar:
    st.text('All Locations')

    if st.button('Beijing City'):
        location_page = 'Beijing'
        homepage = True

    st.text('Locations inside Beijing')

    if st.button('Aotizhongxin'):
        location_page = 'Aotizhongxin'
        homepage = False
        directory = 'daily_dataframes'
        max_aqi_value_one_location, min_aqi_value_one_location, location_max_aqi_one_location, location_min_aqi_one_location = create_highest_lowest_pm25_one_location(directory, location_page)

    if st.button('Dingling'):
        location_page = 'Dingling'
        homepage = False
        directory = 'daily_dataframes'
        max_aqi_value_one_location, min_aqi_value_one_location, location_max_aqi_one_location, location_min_aqi_one_location = create_highest_lowest_pm25_one_location(directory, location_page)
    
    if st.button('Dongsi'):
        location_page = 'Dongsi'
        homepage = False
        directory = 'daily_dataframes'
        max_aqi_value_one_location, min_aqi_value_one_location, location_max_aqi_one_location, location_min_aqi_one_location = create_highest_lowest_pm25_one_location(directory, location_page)
    
    if st.button('Guanyuan'):
        location_page = 'Guanyuan'
        homepage = False
        directory = 'daily_dataframes'
        max_aqi_value_one_location, min_aqi_value_one_location, location_max_aqi_one_location, location_min_aqi_one_location = create_highest_lowest_pm25_one_location(directory, location_page)
    
    if st.button('Gucheng'):
        location_page = 'Gucheng'
        homepage = False
        directory = 'daily_dataframes'
        max_aqi_value_one_location, min_aqi_value_one_location, location_max_aqi_one_location, location_min_aqi_one_location = create_highest_lowest_pm25_one_location(directory, location_page)

    if st.button('Huairou'):
        location_page = 'Huairou'
        homepage = False
        directory = 'daily_dataframes'
        max_aqi_value_one_location, min_aqi_value_one_location, location_max_aqi_one_location, location_min_aqi_one_location = create_highest_lowest_pm25_one_location(directory, location_page)

    if st.button('Nongzhanguan'):
        location_page = 'Nongzhanguan'
        homepage = False
        directory = 'daily_dataframes'
        max_aqi_value_one_location, min_aqi_value_one_location, location_max_aqi_one_location, location_min_aqi_one_location = create_highest_lowest_pm25_one_location(directory, location_page)

    if st.button('Shunyi'):
        location_page = 'Shunyi'
        homepage = False
        directory = 'daily_dataframes'
        max_aqi_value_one_location, min_aqi_value_one_location, location_max_aqi_one_location, location_min_aqi_one_location = create_highest_lowest_pm25_one_location(directory, location_page)
    
    if st.button('Tiantan'):
        location_page = 'Tiantan'
        homepage = False
        directory = 'daily_dataframes'
        max_aqi_value_one_location, min_aqi_value_one_location, location_max_aqi_one_location, location_min_aqi_one_location = create_highest_lowest_pm25_one_location(directory, location_page)

    if st.button('Wanliu'):
        location_page = 'Wanliu'
        homepage = False
        directory = 'daily_dataframes'
        max_aqi_value_one_location, min_aqi_value_one_location, location_max_aqi_one_location, location_min_aqi_one_location = create_highest_lowest_pm25_one_location(directory, location_page)

    if st.button('Wanshouxigong'):
        location_page = 'Wanshouxigong'
        homepage = False
        directory = 'daily_dataframes'
        max_aqi_value_one_location, min_aqi_value_one_location, location_max_aqi_one_location, location_min_aqi_one_location = create_highest_lowest_pm25_one_location(directory, location_page)

if homepage:
    st.title(f'{location_page} PM 2.5 Air Quality')

    # Define the titles and values
    plot_title_Highest_AQI_Value = 'Highest AQI Value'
    plot_title_Lowest_AQI_Value = 'Lowest AQI Value'

    # Create subplots with 1 row and 2 columns, sharing y-axis ticks and labels
    fig, axs = plt.subplots(1, 2, figsize=(8, 2), facecolor='#808080')

    # Plot for the highest AQI value
    axs[0].set_facecolor("#FFF8DC")
    axs[0].text(x=0.5, y=0.8, s=plot_title_Highest_AQI_Value, fontdict={'fontsize': 15, 'color': '#00008B', 'fontfamily': 'sans-serif'}, horizontalalignment='center')
    axs[0].text(x=0.5, y=0.45, s=int(max_aqi_value.round(0)), fontdict={'fontsize': 20, 'color': '#00008B', 'fontfamily': 'sans-serif'}, horizontalalignment='center')
    axs[0].text(x=0.5, y=0.25, s=location_max_aqi, fontdict={'fontsize': 15, 'color': '#00008B', 'fontfamily': 'sans-serif'}, horizontalalignment='center')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # Plot for the lowest AQI value
    axs[1].set_facecolor("#FFF8DC")
    axs[1].text(x=0.5, y=0.8, s=plot_title_Lowest_AQI_Value, fontdict={'fontsize': 15, 'color': '#00008B', 'fontfamily': 'sans-serif'}, horizontalalignment='center')
    axs[1].text(x=0.5, y=0.45, s=int(min_aqi_value.round(0)), fontdict={'fontsize': 20, 'color': '#00008B', 'fontfamily': 'sans-serif'}, horizontalalignment='center')
    axs[1].text(x=0.5, y=0.25, s=location_min_aqi, fontdict={'fontsize': 15, 'color': '#00008B', 'fontfamily': 'sans-serif'}, horizontalalignment='center')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    st.pyplot(fig)

    # Define colors: top 3 in blue, others in gray
    colors = ['blue' if i >= 2 else 'gray' for i in range(5)]

    # Plotting
    fig = plt.figure(figsize=(10, 6))
    bars = plt.barh(avg_pm25_aqi_df['Location'], avg_pm25_aqi_df['Average PM2.5 AQI'], color=colors)

    # Add title and labels
    plt.title('Top 5 Beijing Locations with Highest Average AQI in 2015', fontsize=16)
    plt.xlabel('AQI Average', fontsize=12)

    # Optional: Customize tick parameters
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Set x-axis limit to add padding (increase upper limit by 10% for example)
    plt.xlim(0, avg_pm25_aqi_df['Average PM2.5 AQI'].max() * 1.1)

    # Display values on bars
    for i, bar in enumerate(bars):
        if i >= 2:  # Display values for top 3 bars only
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f}', va='center', fontsize=10)

    # Show the plot
    st.pyplot(fig)

    data = {
        'Location': ['Aotizhongxin', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng',
                    'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong'],
        'Latitude': [40.0455, 40.0154, 39.9276, 39.9334, 39.9381,
                    40.3157, 39.9139, 40.1307, 39.8783, 39.9980, 39.9040],
        'Longitude': [116.3544, 116.2091, 116.4173, 116.3645, 116.1743,
                    116.6311, 116.4735, 116.6598, 116.4072, 116.2953, 116.3517],
    }

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        data, geometry=[Point(xy) for xy in zip(data['Longitude'], data['Latitude'])],
        crs="EPSG:4326"  # WGS 84
    )

    # Add Avg_PM2.5_AQI from the dictionary
    avg_pm25_aqi = {
        'Aotizhongxin': 138.51130418032733,
        'Dingling': 120.33864616577146,
        'Dongsi': 140.44826363472154,
        'Guanyuan': 138.46879329790914,
        'Gucheng': 139.13677174454645,
        'Huairou': 131.0200885635793,
        'Nongzhanguan': 139.9759534542078,
        'Shunyi': 141.3750785245978,
        'Tiantan': 139.02574095457804,
        'Wanliu': 137.02572872265483,
        'Wanshouxigong': 138.3050734726849
    }

    # Map the Avg_PM2.5_AQI values to the GeoDataFrame
    gdf['Avg_PM2.5_AQI'] = gdf['Location'].map(avg_pm25_aqi)

    # Define color map
    cmap = plt.get_cmap('OrRd')
    norm = plt.Normalize(gdf['Avg_PM2.5_AQI'].min(), gdf['Avg_PM2.5_AQI'].max())

    # Plot Beijing and the locations with Avg_PM2.5_AQI
    fig, ax = plt.subplots(figsize=(12, 12))
    gdf.plot(ax=ax, marker='o', column='Avg_PM2.5_AQI', cmap=cmap, markersize=50, alpha=0.6, legend=False, edgecolor='k')

    # Set axis limits with specific bounds
    ax.set_xlim(115.9, 116.8)  # Longitude bounds
    ax.set_ylim(39.7, 40.4)    # Latitude bounds

    # Add labels for the locations with padding
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf['Location']):
        ax.text(x, y + 0.005, label, fontsize=9, ha='right', va='bottom')

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)

    # Add a basemap
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

    plt.title('Average PM2.5 AQI in Beijing Locations (2015)')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    st.pyplot(fig)

    tab1, tab2 = st.tabs(["Daily Data", "Monthly Data"])

    with tab1:
        # Initialize the figure
        fig = plt.figure(figsize=(20, 8))

        # Find the location with the highest overall AQI
        max_aqi = -1
        max_location = None

        for location, df in daily_dataframes.items():
            # Determine the maximum AQI value and corresponding location
            max_aqi_location = df['PM2.5 AQI'].max()
            if max_aqi_location > max_aqi:
                max_aqi = max_aqi_location
                max_location = location

        # Loop through each dataframe in the dictionary
        for location, df in daily_dataframes.items():
            # Determine color: blue for the highest AQI location, gray for others
            color = 'blue' if location == max_location else 'gray'

            # Plot the PM2.5 AQI over time
            plt.plot(df['date'], df['PM2.5 AQI'], label=location, color=color, linewidth=2 if location == max_location else 1)

            # Extract month from the date
            df['month'] = df['date'].dt.month

            # Convert month numbers to abbreviated names
            df['month_abbr'] = df['date'].dt.strftime('%b')

            # Group by month to find the average PM2.5 AQI
            monthly_avg = df.groupby('month_abbr')['PM2.5 AQI'].mean()

            # Find the month with the highest AQI
            highest_month_abbr = monthly_avg.idxmax()

            # Highlight the highest AQI month for each location
            highest_month_data = df[df['month_abbr'] == highest_month_abbr]
            plt.axvspan(highest_month_data['date'].min(), highest_month_data['date'].max(), color='red', alpha=0.2)

        # Extend the x-axis limit (optional: add some padding to the time range)
        plt.xlim(pd.Timestamp('2015-01-01'), pd.Timestamp('2015-12-31') + pd.DateOffset(days=50))

        # Set the x-axis major ticks to display abbreviated month names
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))

        # Add labels and title
        plt.title('PM2.5 AQI Over Time by Location', fontsize=16)

        # Add a legend
        plt.legend(loc='upper right')

        # Show the plot
        st.pyplot(fig)

    with tab2:
        # Initialize the figure
        fig = plt.figure(figsize=(20, 8))

        # Find the location with the highest overall AQI after monthly aggregation
        max_aqi = -1
        max_location = None

        # Dictionary to store the monthly aggregated data
        monthly_aggregated_data = {}

        # Loop through each dataframe in the dictionary to aggregate by month
        for location, df in daily_dataframes.items():
            # Aggregate by month (calculate monthly mean)
            df['month'] = df['date'].dt.to_period('M')
            monthly_avg = df.groupby('month')['PM2.5 AQI'].mean().reset_index()
            monthly_avg['month'] = monthly_avg['month'].dt.to_timestamp()

            # Convert month column to abbreviated month names
            monthly_avg['month'] = monthly_avg['month'].dt.strftime('%b')

            # Store the aggregated data
            monthly_aggregated_data[location] = monthly_avg

            # Determine the maximum AQI value and corresponding location
            max_aqi_location = monthly_avg['PM2.5 AQI'].max()
            if max_aqi_location > max_aqi:
                max_aqi = max_aqi_location
                max_location = location

        # Loop through the monthly aggregated data for plotting
        for location, monthly_df in monthly_aggregated_data.items():
            # Determine color: blue for the highest AQI location, gray for others
            color = 'blue' if location == max_location else 'gray'

            # Plot the PM2.5 AQI aggregated by month
            plt.plot(monthly_df['month'], monthly_df['PM2.5 AQI'], label=location, color=color, linewidth=2 if location == max_location else 1)

            # Highlight the highest AQI month for each location
            highest_month = monthly_df.loc[monthly_df['PM2.5 AQI'].idxmax()]
            plt.axvspan(highest_month['month'], highest_month['month'], color='red', alpha=0.2)

        # Add labels and title
        plt.title('PM2.5 AQI Over Time by Location', fontsize=16)

        # Add a legend
        plt.legend(loc='upper right')

        # Show the plot
        st.pyplot(fig)

    status_columns = ['Good', 'Hazardous', 'Moderate', 'Unhealthy', 'Unhealthy for Sensitive Groups', 'Very Unhealthy']

    # Read the CSV file
    all_status_df_pivot = pd.read_csv('All_Status/All_status.csv', index_col=0)

    # Sum across all locations for each air quality status
    aggregated_values = all_status_df_pivot[status_columns].sum()

    # Sort the aggregated values in ascending order
    aggregated_values_sorted = aggregated_values.sort_values(ascending=True)

    # Determine the color: blue for the max value, gray for others
    colors = ['blue' if v == aggregated_values_sorted.max() else 'gray' for v in aggregated_values_sorted]

    # Create a vertical bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(aggregated_values_sorted.index, aggregated_values_sorted.values, color=colors)

    # Add a title and labels
    plt.title('PM 2.5 Air Quality Status Count in Beijing', fontsize=16)

    # Show the plot
    st.pyplot(plt)

else:
    st.title(f'{location_page} PM 2.5 Air Quality')

    # Define the titles and values
    plot_title_Highest_AQI_Value = 'Highest AQI Value'
    plot_title_Lowest_AQI_Value = 'Lowest AQI Value'

    # Create subplots with 1 row and 2 columns, sharing y-axis ticks and labels
    fig, axs = plt.subplots(1, 2, figsize=(8, 2), facecolor='#808080')

    # Plot for the highest AQI value
    axs[0].set_facecolor("#FFF8DC")
    axs[0].text(x=0.5, y=0.75, s=plot_title_Highest_AQI_Value, fontdict={'fontsize': 15, 'color': '#00008B', 'fontfamily': 'sans-serif'}, horizontalalignment='center')
    axs[0].text(x=0.5, y=0.35, s=int(max_aqi_value_one_location.round(0)), fontdict={'fontsize': 20, 'color': '#00008B', 'fontfamily': 'sans-serif'}, horizontalalignment='center')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # Plot for the lowest AQI value
    axs[1].set_facecolor("#FFF8DC")
    axs[1].text(x=0.5, y=0.75, s=plot_title_Lowest_AQI_Value, fontdict={'fontsize': 15, 'color': '#00008B', 'fontfamily': 'sans-serif'}, horizontalalignment='center')
    axs[1].text(x=0.5, y=0.35, s=int(min_aqi_value_one_location.round(0)), fontdict={'fontsize': 20, 'color': '#00008B', 'fontfamily': 'sans-serif'}, horizontalalignment='center')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    st.pyplot(fig)

    tab1, tab2 = st.tabs(["Daily Data", "Monthly Data"])

    with tab1:
        # Initialize the figure
        fig = plt.figure(figsize=(20, 8))

        # Find the location with the highest overall AQI
        max_aqi = -1
        max_location = None

        
        # Determine the maximum AQI value and corresponding location
        max_aqi_location = daily_dataframes[location_page]['PM2.5 AQI'].max()
        if max_aqi_location > max_aqi:
            max_aqi = max_aqi_location
        
        color = 'blue'

        plt.plot(daily_dataframes[location_page]['date'], daily_dataframes[location_page]['PM2.5 AQI'], label=location_page, color=color, linewidth=2)

        # Extract month from the date
        daily_dataframes[location_page]['month'] = daily_dataframes[location_page]['date'].dt.month

        # Convert month numbers to abbreviated names
        daily_dataframes[location_page]['month_abbr'] = daily_dataframes[location_page]['date'].dt.strftime('%b')

        # Group by month to find the average PM2.5 AQI
        monthly_avg = daily_dataframes[location_page].groupby('month_abbr')['PM2.5 AQI'].mean()

        # Find the month with the highest AQI
        highest_month_abbr = monthly_avg.idxmax()

        # Highlight the highest AQI month for each location
        highest_month_data = daily_dataframes[location_page][daily_dataframes[location_page]['month_abbr'] == highest_month_abbr]
        plt.axvspan(highest_month_data['date'].min(), highest_month_data['date'].max(), color='red', alpha=0.2)

        # Extend the x-axis limit (optional: add some padding to the time range)
        plt.xlim(pd.Timestamp('2015-01-01'), pd.Timestamp('2015-12-31') + pd.DateOffset(days=50))

        # Set the x-axis major ticks to display abbreviated month names
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))

        # Add labels and title
        plt.title('PM2.5 AQI Over Time by Location', fontsize=16)

        # Add a legend
        plt.legend(loc='upper right')

        # Show the plot
        st.pyplot(fig)

    with tab2:
        # Initialize the figure
        fig = plt.figure(figsize=(20, 8))

        # Find the location with the highest overall AQI after monthly aggregation
        max_aqi = -1
        max_location = None

        # Dictionary to store the monthly aggregated data
        monthly_aggregated_data = {}

        # Loop through each dataframe in the dictionary to aggregate by month
        
        # Aggregate by month (calculate monthly mean)
        daily_dataframes[location_page]['month'] = daily_dataframes[location_page]['date'].dt.to_period('M')
        monthly_avg = daily_dataframes[location_page].groupby('month')['PM2.5 AQI'].mean().reset_index()
        monthly_avg['month'] = monthly_avg['month'].dt.to_timestamp()

        # Convert month column to abbreviated month names
        monthly_avg['month'] = monthly_avg['month'].dt.strftime('%b')

        # Store the aggregated data
        monthly_aggregated_data[location_page] = monthly_avg

        # Determine the maximum AQI value and corresponding location
        max_aqi_location = monthly_avg['PM2.5 AQI'].max()
        if max_aqi_location > max_aqi:
            max_aqi = max_aqi_location

        # Loop through the monthly aggregated data for plotting
        for location, monthly_df in monthly_aggregated_data.items():
            # Determine color: blue for the highest AQI location, gray for others
            color = 'blue'

            # Plot the PM2.5 AQI aggregated by month
            plt.plot(monthly_df['month'], monthly_df['PM2.5 AQI'], label=location, color=color, linewidth=2)

            # Highlight the highest AQI month for each location
            highest_month = monthly_df.loc[monthly_df['PM2.5 AQI'].idxmax()]
            plt.axvspan(highest_month['month'], highest_month['month'], color='red', alpha=0.2)

        # Add labels and title
        plt.title('PM2.5 AQI Over Time by Location', fontsize=16)

        # Add a legend
        plt.legend(loc='upper right')

        # Show the plot
        st.pyplot(fig)

    plot_air_quality_status(location_page)