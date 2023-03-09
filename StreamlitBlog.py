#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import seaborn as sns
df_spotipy = pd.read_csv('smaller_dataset.csv')
df_process = pd.read_csv("smaller_dataset.csv")
# In[2]:


with st.sidebar:
    st.sidebar.header("Welkom bij de Navigatiemenu")
    navigatie = st.selectbox ("Kies de gewenste pagina", 
                              ("Hoofdpagina", "Data Verzameling", 
                               "Data Verkenning", 
                               "Plot 1: Radarplot", 
                               "Plot 2: Kaartenplot", 
                               "Plot 3: Histogram",
                               "Plot 4: Subplot",
                               "Plot 5: Scatterplot",
                               "Recommendations"))
# In[3]:


#De Spotify logo URL aanroepen
response = requests.get("https://cdn.discordapp.com/attachments/1074735154939052173/1082418278821015674/spotify.png")
spot_image = Image.open(BytesIO(response.content))


# In[4]:


if navigatie == "Hoofdpagina":
    st.title("Casus 2: De :green[Spotify] Casus")
    st.subheader("Team 5: Nora Ruijpers, Sam van Doorn, Stijn Maat, Tarik Kiliç")
    st.image(spot_image, caption= "Spotify")

    st.write("Welkom bij de blog. Het internet geeft toegang tot een onvoorstelbaar hoeveelheid data; data een data analist kan gebruiken om analyse uit te voeren")
    st.write("Met de behulp van APIs, kan deze data aangeroepen worden. In deze blog is er analyse uitgevoerd op data van de volgende bronnen:")
    st.markdown("- De Spotify API")
    st.markdown("- De Charts API op Kaggle")
    st.markdown("- De lijst van populairste artiesten sinds het begin van het millenium, gehaald van Chart2000.com")

    st.write("Met behulp van Streamlit zal de grondig verkende data gebruikt worden om de gevonden inzichten te visualiseren.")


# In[5]:


#voor de Data Verzameling Tab
if navigatie == "Data Verzameling":
    
    #Navigatiemenu binnen pagina met anchor links
    st.sidebar.markdown('''
    - [De Spotify API](#de-spotify-api)
    - [De Chart2000 Dataset](#de-chart2000-dataset)
    ''')

    st.title("Data verzameling")
    
    #Spotify API uitleg:
    st.header("[De Spotify API](https://developer.spotify.com/)")
    st.write('''Zoals vermeld op de hoofdpagina, zijn wij aan de slag gegaan met de Spotify API. Spotify is een populaire muziek streaming service gebruikt door miljoenen mensen over de hele wereld.   
    Met de openbare API beschikbaar van Spotify is er data te verzamelen over de artiesten en de liedjes zelf.  
    Van de Spotify API zijn de volgende datapunten verzameld voor gebruik voor de aankomende analyses:      
    ''')

    datacol1, datacol2 = st.columns(2, gap = "medium")
    with datacol1:
        st.subheader("Artiesten")
        st.markdown('''
        - **artist_id** : Een unieke ID voor elke artiest
        - **artist_name**: De naam van de artiest 
        - **artist_popularity**: Een numerieke waarde dat de relatieve populariteit van de artiest weergeeft.
        - **artist_followers**: De hoeveelheid volgers dat de artiest heeft op Spotify
        - **artist_genres**: De muziekgenres waarin de artiest in voorkomt
        ''')
    
    with datacol2:
        st.subheader("Liedjes")
        st.markdown('''
        - **track_name**: Naam van het liedje.
        - **track_duration**: Lengte van het liedje in milliseconde.
        - **track_popularity**: Een numerieke waarde dat de relatieve populariteit van het liedje weergeeft.
        - **track_streams_per_country**: Het aantal keer dat het liedje is beluistert, per land.
        - **collaborations**: boolean waarde om te laten zien of het liedje gezongen is in collaboratie met andere artiest(en).
        - **release_date**: De datum waarop het liedje openbaar beschikbaar werd gemaakt. 
        ''')

        st.write("Spotify verzameld ook zogenoemde 'Audio Features' voor elk lied.")
        
        st.markdown('''
        - **danceability**: Waarde tussen 0 en 1 dat de dansbaarheid van het liedje bepaalt op basis muziekale elementen zoals tempo, regulariteit, en ritme stabiliteit.
        - **energy**: Waarde tussen 0 en 1 dat de intensiteit van het liedje representeert. Liedjes met hogere waardes zijn typisch snelle en luide muziek, met voorbeeld muziek uit de Death Metal genre.            
        - **key**: De toonsoort van het liedje.            
        - **loudness**: De algemene luidheid van het liedje, in decibels. Waardes vallen in het algemeen tussen -60 en 0 dB. 
        - **mode**: De modaliteit van het liedje, boolean waarde waar 0 voor minor staat en 1 voor major.
        - **speechiness**: Waarde tussen 0 en 1 dat de hoeveelheid gesproken woorden in het liedje representeert.
        - **acousticness**: Waarde tussen of gelijk aan 0 en 1 dat de acousticiteit van het liedje representeert.
        - **instrumentalness**: Waarde tussen 0 en 1 dat voorspeld of een liedje wel of geen vocals heeft.
        - **liveness**: Waarde tussen 0 en 1 dat meet of er een publiek aanwezig was tijdens de opname van het liedje.
        - **valence**: Waarde tussen 0 en 1 dat de positiviteit van een liedje meet, waar hogere waardes positiever klinken.
        - **tempo**: De tempo van het liedje in BPM (beats per minute)
        ''')
    
    st.header("[De Chart2000 dataset](https://chart2000.com/)")
    st.write('''
    Er zijn miljoenen artiesten op Spotify. Het zou enorm onpraktisch zijn om analyse uit te voeren op elke artiest op de platform.  Daarom is er met behulp van Chart2000 
    een lijst gemaakt met artiesten die gebruikt zullen worden in de analyse. Van de website zijn de top 200 albums van het jaar 2000, 2010, en 2020 opgehaald. Deze data is gecombineerd, en gefilterd om een lijst met artiesten te krijgen.  
    Tijdens het filteren is er rekening gehouden dat dezelfde artiest niet meer dan één keer in de lijst voorkomt. Ook is er rekening gehouden met albums die zijn gemaakt in collaboratie met andere artiesten.  
    Na het filteren van de in totaal 600 albums is er een lijst van ongeveer 280 artiesten overgebleven. Een stuk minder dan de miljoenen artiesten op Spotify, maar wel groot genoeg om interessante analyse mee uit te voeren.  
    ''')
    
    


# In[6]:


# df_charts = pd.read_csv("charts.csv")
# df_artiestenlijst = pd.read_csv("artiestenlijst.csv")
df_process=pd.read_csv("smaller_dataset.csv")


# In[7]:

if navigatie == "Data Verkenning":
    #Code voor data verkenning
    st.title("Data Verkenning")
    
    #Spotify API uitleg:
    st.header("[De Spotify API](https://developer.spotify.com/)")
    st.write('''
        Het besluit om te werken met de Spotify API leed tot een leuke maar uitdagende casus.   
        De overvloed van keuzes maakt het keuzeprocess ingewikkeld. Uiteindelijk is er besloten om met de Audio Features & Analysis gedeelte van de Spotify API te werken met de doel om relaties te vinden
        tussen de audio features van liedjes en de mensen die het beluisteren.  
        Van de beschikbare data, hebben we data dat niet relevant was voor onze analyses weggelaten. Dit is voornamelijk data dat beschikbaar is voor **Audiobooks en Podcasts** beschikbaar op Spotify.   
        Met de analyses gepland waren wij voornamelijk geinteresseerd in muziek, de inclusie van audiobooks en podcasts zou een ongewenste graad van ingewikkeldheid met zich meebrengen. 
        De resultaat van de dataverkenning heeft geleid tot de volgende dataset:              
    ''')
    # Hier kan dus gewoon de header van de dataframe die we hebben van Spotify API zoals we in Discord hebben besproken. 
    #zet voor nu gewoon een dummy

    

    #End van de Spotify Uitleg als het goed is
    
#     st.dataframe(df_responses)
       
    st.header("[De Chart2000 dataset](https://chart2000.com/)")
    st.write('''
    Vergeleken met de dataverkenning van de andere datasets, was de dataverkenning voor de data van Chart2000 vrij simpel.   
    Deze data is alleen aangeroepen om als input te gebruiken voor analyses met artiesten, alleen de artiestnamen waren nodig.    
    Chart2000 maakt de data beschikbaar via csv bestanden. De csv bestanden voor de top 200 albums van 2000, 2010, en 2020 zijn samengevoegd tot één lijst.   
    De variabelen in de dataset dat niet de artiestnaam was, zoals de albumnaam, top 200 positie per land, en aangegeven winst, zijn verwijderd van de dataset.    
    Artiestnamen die meerdere keren voorkomen zijn verwijderd, voor de doel van de analyse hebben we het maar één keer nodig. Artistnamen met een komma in de artiestnaam zijn gesplits op de komma: dit waren collaboratiealbums.    
    Verder zijn de artiestnamen "Original Soundtrack" en "Original Cast" verwijderd. Deze artiestnamen zijn algemene termen gebruikt voor de sountracks van musicals en films.
    ''')
    
#     st.dataframe(df_artiestenlijst)
    
    st.header("Gecombineerde dataset")
    
    st.write('''
    Voor het verkrijgen van de spotify data zijn de volgende acties ondernomen.       
    Als eerste werd er gezocht naar een lijst met artiesten die kon worden gebruikt om op te zoeken.   
    Daarna gingen we gebruik maken van de spotify web API. Hierbij is het noodzakelijk om een spotify account te hebben om
    toegang tot deze data te hebben.     
    Wanneer je de beschikking hebt van een kan je onder de applicatie deskboard een app aanmaken.     
    Door die app aan te maken krijg je toegang om data te verzamelen van de API door gebruikt te maken van het client id en secret id die terugkomt 
    in het maken van de code.  
    Belangrijk hierbij is dat elk id anders is en dat deze niet gedeelt mag worden. In de app kan worden gezien hoeveel 
    request er worden gedaan en wat voor specifieke variabelen.     
    Door de package spotipy te gebruiken was het mogelijk om de spotify api in combinatie met de artiestenlijst het mogelijk om de data te verzamelen.       
    Daarnaast zijn er nog andere api's gebruikt om meer data te verkrijgen zoals de spotify charts van kaggle waardoor er bepaalde kolommen konden worden toegevoegd voor nog meer precieze data.            
    Later is ook nog gebruik gemaakt van een spotify dataset van songs in de periode 1921-2020 om een vergelijking te maken tussen beide 
    datasets en er conclusies uit te trekken.      
    Voor het maken van voorspellingen is er een song recommendation gemaakt die bepaald welke nummers mogelijk bij jouw passen aan aanleiding van een geselecteerd nummer.         
    ''')
    st.dataframe(df_process)


# In[8]:


if navigatie == "Plot 1: Radarplot":
    #Code voor de eerste plot
    st.title("Plot 1: [Radar plot - Vergelijking van 2 artiesten]")

    df_process['track_duration'] = df_process['track_duration'] / 60000

    # Get a list of all artists
    artist_list = df_process['artist_name'].unique().tolist()

    # Allow the user to select one or more artists
    selected_artists = st.multiselect('Select one or more artists', artist_list)

    # Define the audio feature columns
    audio_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'track_duration']

    # Create the polar plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")

    # Define the angles for each feature
    theta = np.linspace(0, 2*np.pi, len(audio_cols), endpoint=False)

    # Plot each selected artist's audio features
    for artist in selected_artists:
        # Get a list of all songs by the selected artist
        song_list = df_process[df_process['artist_name'] == artist]['track_name'].unique().tolist()

        # Allow the user to select a song by the selected artist
        selected_song = st.selectbox(f'Select a song by {artist}', song_list)

        # Filter the dataset to only include the selected song
        df_selected = df_process[(df_process['artist_name'] == artist) & (df_process['track_name'] == selected_song)]

        # Get the mean value for each audio feature
        values = df_selected[audio_cols].mean().values

        # Draw the polygon and fill it with the selected color
        ax.plot(theta, values, marker="o", label=artist)
        opacity = st.slider(f'Opacity for {artist}', 0.0, 1.0, 0.1, 0.1)
        ax.fill(theta, values, alpha=opacity)

    # Set the title of the plot
    plt.title(f"Audio Features for Selected Songs")

    # Set the axis labels to display vertically
    plt.gcf().axes[0].set_thetagrids(np.degrees(theta), audio_cols, fontsize=12, ha='center', va='center')

    # Display the plot
    st.pyplot(fig)


# In[9]:


if navigatie == "Plot 2: Kaartenplot":
    st.title("Plot 2: [Kaartenplot, Average Popularity of Tracks by Country ]")
    
    # Group the data by country and calculate the mean popularity for artists and tracks
    grouped_data = df_process.groupby('country')['artist_popularity'].mean().reset_index()
    grouped_data1 = df_process.groupby('country')['track_popularity'].mean().reset_index()
    grouped_data2 = df_process.groupby('country')[['artist_name','artist_popularity']].mean().reset_index()
    grouped_data3 = df_process.groupby('country')[['track_name','track_popularity']].mean().reset_index()

    # Define a function to plot choropleth maps based on user selection
    def plot(checkbox_1, checkbox_2, artist_selectbox):
        fig = go.Figure()
    
        if checkbox_1:
            fig.add_trace(px.choropleth(grouped_data, locations='country', locationmode='country names',
                                    color='artist_popularity', range_color=[85, 90],
                                    color_continuous_scale='viridis',
                                    title='Average Popularity of Artists by Country').data[0])
        if checkbox_2:
            fig.add_trace(px.choropleth(grouped_data1, locations='country', locationmode='country names',
                                    color='track_popularity', range_color=[73, 80],
                                    color_continuous_scale='viridis',
                                    title='Average Popularity of Tracks by Country').data[0])
    
        if artist_selectbox:
            filtered_data = df_process[df_process['artist_name']==artist_selectbox].groupby('country')['track_popularity'].mean().reset_index()
            fig.add_trace(px.choropleth(filtered_data, locations='country', locationmode='country names',
                                    color='track_popularity', range_color=[73, 80],
                                    color_continuous_scale='viridis',
                                    title=f'Average Popularity of Tracks by {artist_selectbox} by Country').data[0])
    
        st.plotly_chart(fig)

    # Define sidebar widgets for user selection
    checkbox_1 = st.sidebar.checkbox('Average Popularity of Artists by Country', key='checkbox_1')
    checkbox_2 = st.sidebar.checkbox('Average Popularity of Tracks by Country', key='checkbox_2')
    artist_selectbox = st.sidebar.selectbox("Select an artist to display", options=df_process["artist_name"].unique())

    # Call the plot function based on user selection
    if checkbox_1 or checkbox_2 or artist_selectbox:
        plot(checkbox_1, checkbox_2, artist_selectbox)
    else:
        st.write('Please select a valid option.')
    
    
    
    
    
    #Code voor de tweede plot
    

    # Assuming the data is stored in a DataFrame called 'df_process'
    #grouped_data = df_process.groupby('country')['artist_popularity'].mean().reset_index()
    #grouped_data2 = df_process.groupby('country')['track_popularity'].mean().reset_index()
    #options = st.sidebar.radio("Select:" ,("Average Popularity of Artists by Country","Average Popularity of Tracks by Country"))
    #def plot(options):
     #   if options == "Average Popularity of Artists by Country":
      #      fig = px.choropleth(grouped_data, locations='country', locationmode='country names',
       #                         color='artist_popularity', range_color=[85, 90], 
        #                        color_continuous_scale='viridis',
         #                       title='Average Popularity of Artists by Country')
       # if options == "Average Popularity of Tracks by Country":
        #    fig = px.choropleth(grouped_data2, locations='country', locationmode='country names',
         #                       color='track_popularity', range_color=[73, 80], 
          #                      color_continuous_scale='viridis',
           #                     title='Average Popularity of Tracks by Country')
        #return fig
    #st.plotly_chart(fig)

if navigatie == "Plot 3: Histogram":
    #Code voor de dittes
    st.title("Plot 3: Histogram")
 
    # Define genre options and default selection
    genre_options = ['pop', 'chicagorap', 'dancepop', 'atlhiphop', 'barbadianpop', 'detroithiphop', 'artpop', 'canadianpop', 'canadiancontemporaryr&b', 'kpop', 'conscioushiphop', 'bigroom', 'canadianhiphop', 'colombianpop', 'reggaeton']
    default_genres = ['pop', 'dancepop']

    # Create checkbox for genre selection
    selected_genres = st.sidebar.multiselect('Select Genres', genre_options, default=default_genres)

    # Filter data by selected genres
    df_filtered = df_process[df_process['artist_genres'].str.contains('|'.join(selected_genres))]

    # Extract artist genres and count frequency
    genres_counts = df_filtered['artist_genres'].str.split(';').explode().value_counts()

    # Create histogram using Matplotlib
    fig, ax = plt.subplots()
    colors = plt.cm.Set2(range(len(genres_counts)))
    for i, (genre, count) in enumerate(genres_counts.iteritems()):
        if genre in selected_genres:
            ax.hist(count, bins=20, color=colors[i], label=genre)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Number of Genres")
    ax.set_title("Distribution of Selected Artist Genres")
    ax.legend()
    st.pyplot(fig)

#     # Define genre options and default selection
#     genre_options = ['pop', 'chicagorap', 'dancepop', 'atlhiphop', 'barbadianpop', 'detroithiphop', 'artpop', 'canadianpop', 'canadiancontemporaryr&b', 'kpop', 'conscioushiphop', 'bigroom', 'canadianhiphop', 'colombianpop', 'reggaeton']
#     default_genres = ['pop', 'dancepop']

#     # Create checkbox for genre selection
#     selected_genres = st.sidebar.multiselect('Select Genres', genre_options, default=default_genres)

#     # Filter data by selected genres
#     df_filtered = df_process[df_process['artist_genres'].str.contains('|'.join(selected_genres))]

#     # Extract artist genres and count frequency
#     genres_counts = df_filtered['artist_genres'].str.split(';').explode().value_counts()

#     # Create histogram using Matplotlib
#     fig, ax = plt.subplots()
#     ax.hist(genres_counts, bins=20)
#     ax.set_xlabel("Frequency")
#     ax.set_ylabel("Number of Genres")
#     ax.set_title("Distribution of Selected Artist Genres")
#     st.pyplot(fig)

if navigatie == "Plot 4: Subplot":
    #Code voor de dattes
    st.title("Plot 4: Subplot")
    
    fig, ax = plt.subplots(figsize=(12, 10))

    # Compute popularity and plot bar chart using Seaborn
    lead_artists = df_process.groupby('artist_name')['artist_popularity'].sum().sort_values(ascending=False).head(20)
    sns.barplot(x=lead_artists.values, y=lead_artists.index, palette="Greens", orient="h", edgecolor='black', ax=ax)

    # Customize plot labels and title
    ax.set_xlabel('Sum of Popularity', c='r', fontsize=12)
    ax.set_ylabel('Artist', c='r', fontsize=12)
    ax.set_title('20 Most Popular Artists in Dataset', c='r', fontsize=14, weight='bold')

    # Show plot using Streamlit
    st.pyplot(fig)

if navigatie == "Plot 5: Scatterplot":
    #Code voor de dit dan maar
    st.title("Plot 5: Scatterplot - song streams vs tempo")
    
    fig = px.scatter(df_spotipy, x= 'artist_popularity', y = 'artist_followers', color = 'artist_name', size = 'artist_followers')
    fig.update_layout(title="Popularity over Followers",xaxis= dict(title = "Popularity Score [0-100]"))
    fig.show()
    st.plotly_chart(fig)
    
    #fig, ax = plt.subplots()
    #ax.scatter(df_process['position'], df_process['chart_trend'], alpha=0.5)
    #ax.set_xlabel("Position")
    #ax.set_ylabel("Chart_trend")
    #ax.set_title("Correlation between Position and Chart Trend")
    #st.pyplot(fig)
    
if navigatie == "Recommendations":
    #Code voor de recommendations
    st.title("Recommendations")
    st.write('''Het gebruik van deze functie gaat als volgt:     
    Door een track van de dataset in te vullen wordt er een prediction gemaakt voor welke nummers vergelijkbaar zijn met dit nummer.     
    Hou er rekening mee dat deze applicatie hoofdlettergevoelig is en dat de titel compleet moet zijn voor een goede voorspelling.   
    ''')
    # Load the smaller Spotify dataset
    df_spotify = pd.read_csv('smaller_dataset.csv')

    # Select only the columns that you want to keep in the smaller dataset
    columns_to_keep = ['artist_name', 'track_name', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                    'instrumentalness', 'liveness', 'valence', 'tempo']

    # Create the smaller dataset by selecting only the rows and columns that you want to keep
    df_smaller = df_spotify[columns_to_keep].copy()

    # Remove any rows with missing values
    df_smaller.dropna(inplace=True)

    # Reset the index of the dataframe
    df_smaller.reset_index(drop=True, inplace=True)

    # Define a function to get song recommendations
    def get_song_recommendations(song_name, N=10):
        
        # Get the song features for the given song name
        song = df_smaller[df_smaller['track_name'] == song_name].iloc[0]
        song_features = song[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                             'instrumentalness', 'liveness', 'valence', 'tempo']].values.reshape(1, -1)

        # Calculate the distance between the given song and all other songs in the dataset
        distances = pairwise_distances(df_smaller[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                                                 'instrumentalness', 'liveness', 'valence', 'tempo']].values,
                                       song_features,
                                       metric='euclidean').flatten()

        # Get the indices of the N songs with the smallest distance, excluding the index of the given song
        song_index = df_smaller[df_smaller['track_name'] == song_name].index[0]
        indices = distances.argsort()[1:N+1]
        indices = [i for i in indices if i != song_index]

        # Get the track names of the recommended songs
        recommendations = df_smaller.iloc[indices][['artist_name', 'track_name']].values.tolist()
        return recommendations

    song_input = st.text_input("Enter your song")
    st.markdown(f"Your input is: {song_input}")

    # Example usage
    if song_input:
        original_song = df_smaller[df_smaller['track_name'] == song_input].iloc[0]
        artist_name = original_song['artist_name']
        recommendations = get_song_recommendations(song_input, N=10)
        st.write(f'Song recommendations for "{song_input} - {artist_name}":')
        for i, song in enumerate(recommendations):
            st.write(f'{i+1}. {song}')

    st.write('''Voor het maken van deze aplicatie zijn de volgende handelingen getroffen.   
    Als eerst is het functie 'pairwise_distances' geinstalleerd vanuit de package scikit-learn.    
    Deze functie wordt meestal gebruikt om bewerkingen toe te passen bij matrixen en arrays.   
    Om gebruik hiervan te maken worden de volgende acties ondernomen. 
    Als eerste wordt er een gebruik gemaakt van een kleiner dataframe dat is gebasseerd op de top 100 van de voorgaande dataset.    
    Daarna zijn alle NaN values verwijderd uit het dataframe en is de reset_index gebruikt om een new dataframe te maken.  
    Als tweede is er de functie 'get_song_recommendations' om 10 verschillende songs terug te geven aan de gebruiker die overeenkomen met het door de gebruiker opgegeven nummer.   
    
    
    
    
    
    
     ''')

    import subprocess

    # Get the list of installed packages
    pip_list = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
    installed_packages = [line.split("==")[0] for line in pip_list.split("\n")]

    # Write the list of installed packages to a requirements file
    with open("requirements.txt", "w") as f:
        f.write("\n".join(installed_packages))
