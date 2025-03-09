import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

# Load data
day_df = pd.read_csv('dashboard/day_cleaned.csv')
hour_df = pd.read_csv('dashboard/hour_cleaned.csv')

# Konversi kolom 'dteday' ke datetime64
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])

min_date = day_df["dteday"].min()
max_date = day_df["dteday"].max()

# Sidebar
with st.sidebar:
    # Menentukan default value sebagai rentang yang valid
    with st.form(key='date_range'):
        date_range = st.date_input(
            label='Rentang Waktu',
            min_value=min_date.date(),
            max_value=max_date.date(),
            value=(min_date.date(), max_date.date()) if min_date != max_date else (min_date.date(), min_date.date())
        )
        submit_button = st.form_submit_button(label='Submit')
    
    if len(date_range) == 1:
        st.warning("Harap pilih rentang tanggal yang valid (dua tanggal)!")
        st.stop()


# Konversi ke datetime64
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

# Filter data
filtered_day_df = day_df[(day_df['dteday'] >= start_date) & (day_df['dteday'] <= end_date)]
filtered_hour_df = hour_df[(hour_df['dteday'] >= start_date) & (hour_df['dteday'] <= end_date)]


# Helper function yang dibutuhkan untuk menyiapkan berbagai dataframe
def create_grouped_by_year_month_df(df: pd.DataFrame) -> pd.DataFrame:
    grouped_df = df.groupby(['yr', 'mnth'])['cnt'].sum().unstack(level=0)
    year_labels = {i: 2011 + i for i in grouped_df.columns}
    grouped_df.rename(columns=year_labels, inplace=True)
    
    return grouped_df

def create_grouped_by_season_df(df: pd.DataFrame) -> pd.DataFrame:
    grouped_df = df.groupby('season').agg({
        'casual': ['mean'],
        'registered': ['mean'],
    })

    grouped_df.rename(index={1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}, inplace=True)

    return grouped_df

def create_grouped_by_hour_df(df: pd.DataFrame) -> pd.DataFrame:
    grouped_df = df.groupby('hr').cnt.mean()
    
    return grouped_df

def create_grouped_categorized_temp_hum_ws(df: pd.DataFrame) -> pd.DataFrame:
    # Definisikan interval kategori untuk suhu, kelembaban, dan kecepatan angin
    bins_temp = np.linspace(df['temp'].min(), df['temp'].max(), 6)  # 5 kategori suhu
    bins_atemp = np.linspace(df['atemp'].min(), df['atemp'].max(), 6)  # 5 kategori atemp
    bins_hum = np.linspace(df['hum'].min(), df['hum'].max(), 6)  # 5 kategori kelembaban
    bins_wind = np.linspace(df['windspeed'].min(), df['windspeed'].max(), 6)  # 5 kategori angin

    # Buat label kategori
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

    # Konversi nilai menjadi kategori (binning)
    df['temp_group'] = pd.cut(df['temp'], bins=bins_temp, labels=labels)
    df['atemp_group'] = pd.cut(df['atemp'], bins=bins_atemp, labels=labels)
    df['hum_group'] = pd.cut(df['hum'], bins=bins_hum, labels=labels)
    df['wind_group'] = pd.cut(df['windspeed'], bins=bins_wind, labels=labels)

    # Group by berdasarkan kategori dan hitung rata-rata penyewaan sepeda
    temp_effect = df.groupby('temp_group')['cnt'].mean()
    atemp_effect = df.groupby('atemp_group')['cnt'].mean()
    hum_effect = df.groupby('hum_group')['cnt'].mean()
    wind_effect = df.groupby('wind_group')['cnt'].mean()

    # Gabungkan hasil dalam satu DataFrame
    effect_df = pd.DataFrame({
        'Temperature (temp)': temp_effect,
        'Feeling Temperature (atemp)': atemp_effect,
        'Humidity (hum)': hum_effect,
        'Wind Speed (windspeed)': wind_effect
    })
    
    return effect_df

# Menyiapkan dataframe
day_grouped_by_year_month_df = create_grouped_by_year_month_df(day_df)
day_grouped_by_season_df = create_grouped_by_season_df(filtered_day_df)
hour_grouped_df = create_grouped_by_hour_df(filtered_hour_df)
categorized_effect_df = create_grouped_categorized_temp_hum_ws(day_df)

# Menampilkan judul
st.title('Bike Sharing Analysis Dashboard')


# Menampilkan plot
# =============================================================
st.subheader('Bike Rent Trend')
figure = plt.figure(figsize=(12, 6))
for year in day_grouped_by_year_month_df.columns:
    plt.plot(day_grouped_by_year_month_df.index, day_grouped_by_year_month_df[year], marker='o', linestyle='-', label=str(year))
plt.title('Bike Rent Trend by Year and Month')
plt.xlabel('Month')
plt.ylabel('Total Bike Rented')
plt.xticks(np.arange(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Year')
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
st.pyplot(figure)

# =============================================================
st.subheader('Bike Rent Average by Hour')
figure = plt.figure(figsize=(12, 6))
max_hour = hour_grouped_df.idxmax()
bars = plt.bar(hour_grouped_df.index, hour_grouped_df.values, color='#a7d8d2', width=0.7)
bars[max_hour].set_color('#5e9ecf')
plt.title("Bike Rent Average by Hour")
plt.xlabel("Hour")
plt.ylabel("Bike Rent Average")
# Menampilkan semua label di sumbu X
plt.xticks(hour_grouped_df.index)

st.pyplot(figure)

# =============================================================
st.subheader('Bike Rent Average by Season')
figure = plt.figure(figsize=(12, 6))
casual = day_grouped_by_season_df['casual']['mean']
registered = day_grouped_by_season_df['registered']['mean']
total = casual + registered  # Total penyewaan per musim
# Menentukan musim dengan total penyewaan terbesar
max_season = total.idxmax()
# Warna utama untuk musim terbesar
color_casual_main = '#5e9ecf'
color_registered_main = '#73b97c'
# Warna pudar untuk musim lainnya
color_casual_fade = '#a7d8d2'
color_registered_fade = '#bfe3b2'
# Menentukan warna setiap musim
colors_casual = [color_casual_main if season == max_season else color_casual_fade for season in total.index]
colors_registered = [color_registered_main if season == max_season else color_registered_fade for season in total.index]
# Plot
seasons = total.index
x = np.arange(len(seasons))
plt.bar(x, casual, label='Casual', color=colors_casual)
plt.bar(x, registered, bottom=casual, label='Registered', color=colors_registered)
plt.xticks(x, seasons)
plt.xlabel('Season')
plt.ylabel('Bike Rent Average per Day')
plt.title('Bike Rent Average by Season')
plt.legend(
    [
        plt.Rectangle((0, 0), 1, 1, color=color_casual_main),
        plt.Rectangle((0, 0), 1, 1, color=color_registered_main)
    ],
    ['Casual', 'Registered'],
    loc='upper right',
    fontsize=10,
    facecolor='white',
    framealpha=1,
    edgecolor='black'
)

st.pyplot(figure)

# =============================================================
st.subheader('Effect of Temperature, Feeling Temperature, Humidity, and Windspeed on Bike Rent Total')
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.regplot(ax=axes[0, 0], x=day_df['temp'], y=day_df['cnt'], scatter_kws={'color': 'skyblue'}, line_kws={'color': 'blue'})
axes[0, 0].set_title("Temperature vs Bike Rent Total")
axes[0, 0].set_xlabel("Temperature")
axes[0, 0].set_ylabel("Bike Rent Total")

sns.regplot(ax=axes[0, 1], x=day_df['atemp'], y=day_df['cnt'], scatter_kws={'color': 'skyblue'}, line_kws={'color': 'blue'})
axes[0, 1].set_title("Feeling Temperature vs Bike Rent Total")
axes[0, 1].set_xlabel("Feeling Temperature")
axes[0, 1].set_ylabel("Bike Rent Total")

sns.regplot(ax=axes[1, 0], x=day_df['hum'], y=day_df['cnt'], scatter_kws={'color': 'skyblue'}, line_kws={'color': 'blue'})
axes[1, 0].set_title("Humidity vs Bike Rent Total")
axes[1, 0].set_xlabel("Humidity")
axes[1, 0].set_ylabel("Bike Rent Total")

sns.regplot(ax=axes[1, 1], x=day_df['windspeed'], y=day_df['cnt'], scatter_kws={'color': 'skyblue'}, line_kws={'color': 'blue'})
axes[1, 1].set_title("Wind Speed vs Bike Rent Total")
axes[1, 1].set_xlabel("Wind Speed")
axes[1, 1].set_ylabel("Bike Rent Total")

plt.tight_layout()
st.pyplot(fig)

# =============================================================
st.subheader('Effect of Categorized Temperature, Feeling Temperature, Humidity, and Windspeed on Bike Rent Average')
figure = plt.figure(figsize=(12, 6))

# Plot setiap kategori dengan garis berbeda
plt.plot(categorized_effect_df.index, categorized_effect_df['Temperature (temp)'], marker='o', label='Temperature', color='#1f77b4')
plt.plot(categorized_effect_df.index, categorized_effect_df['Feeling Temperature (atemp)'], marker='s', label='Feeling Temperature', color='#ff7f0e')
plt.plot(categorized_effect_df.index, categorized_effect_df['Humidity (hum)'], marker='^', label='Humidity', color='#2ca02c')
plt.plot(categorized_effect_df.index, categorized_effect_df['Wind Speed (windspeed)'], marker='d', label='Wind Speed', color='#d62728')

# Tambahkan label dan judul
plt.xlabel("Category")
plt.ylabel("Bike Rent Average")
plt.title("Bike Rent Average by Categorized Temperature, Humidity, and Windspeed")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

st.pyplot(figure)

