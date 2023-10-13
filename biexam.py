import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn import metrics
import folium
from streamlit_folium import folium_static
if 'predict_cluster_pressed' not in st.session_state:
    st.session_state.predict_cluster_pressed = False

# Load datasets
epl_standings = pd.read_csv('/Users/aleksbuha/Desktop/BIExam/data/EPLStandings2000-2022.csv')
merged_data = pd.read_excel('/Users/aleksbuha/Desktop/BIExam/data/playerTransfers2000-2022.xlsx')

# Sidebar menu
menu = ['Homepage', 'Data Exploration and Visualization', 'Supportive Dataset', 'Multiple Linear Regression', 'Classifications', 'K-means', 'Geo Map', 'Summary']
section = st.sidebar.selectbox("Menu", menu)

# Homepage
if section == 'Homepage':
    st.title("Premier League Analysis")
    st.write("This project provides an analysis of the English Premier League from 2000 to 2022. Navigate through the sections to explore various visualizations, models, predictions and insights derived from the data.")
    st.subheader("Background Information")
    st.write("The English Premier League (EPL) is one of the most watched football leagues in the world. It attracts top talent and has a broad global fanbase. Its competitive nature means that matches are unpredictable, adding to the excitement and fan engagement.")
    st.subheader("Project Motivation")
    st.write("In today's data-driven world, there's a growing interest in sports analytics. For the EPL, predicting team performances can benefit various stakeholders. Fans might be interested for the sake of discussion or for making decisions in fantasy football leagues. Analysts and pundits can use predictive insights to enhance their commentary and post-match discussions. Moreover, with the rise of sports betting, accurate predictions can be valuable for those looking to place informed bets.")
    st.subheader("Objective")
    st.write("The primary aim of this project is to utilize historical data, player transfer information, and other relevant metrics to predict Premier League team performances. This will involve analyzing past seasons, understanding patterns, and building predictive models to foresee future team standings and performances.")
    
# Load and display an image
    image_path = "/Users/aleksbuha/Desktop/BIExam/Images/homepage.png"
    st.image(image_path, use_column_width=True)

# Data Exploration and Visualization
elif section == 'Data Exploration and Visualization':
    st.title("Data Exploration and Visualization")
    st.write("Overview of the EPL standings dataset. First five rows of the dataset")
    st.write(epl_standings.head())
    
    # Visualization: Team's performance over the years
    st.write("Team's performance over the years")
    team = st.selectbox("Select a team", epl_standings['Team'].unique())
    team_data = epl_standings[epl_standings['Team'] == team]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(team_data['Season'], team_data['Pts'], marker='o')
    ax.set_title(f"{team}'s Performance (2000-2022)")
    ax.set_xlabel('Season')
    ax.set_ylabel('Points')
    ax.grid(True)
    ax.set_xticks(team_data['Season'])
    ax.set_xticklabels(team_data['Season'], rotation=45)
    st.pyplot(fig)
    
    # Histogram: Distribution of Points in EPL Standings over seasons
    st.write("Histogram: Distribution of Points in EPL Standings over Seasons")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=epl_standings, x='Pts', kde=True, bins=30, ax=ax)
    ax.set_title('Histogram: Distribution of Points in EPL Standings')
    st.pyplot(fig)
    st.write("This visualization provides a clear snapshot of how often teams achieve certain points brackets, giving insight into the most common performance levels in the league. By understanding the distribution of points, we can better predict the potential end-of-season position of teams, as the accumulation of points directly correlates with league standings.")

    # Box Plot: Distribution of Goals For and Goals Against
    st.write("Box Plot: Distribution of Goals For and Goals Against")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxenplot(data=epl_standings, y='GF', ax=ax)
    sns.boxenplot(data=epl_standings, y='GA', color='red', ax=ax)
    ax.set_title('Box Plot: Distribution of Goals For (Blue) and Goals Against (Red)')
    st.pyplot(fig)
    st.write("The box plot offers a clear comparison between the distribution of 'Goals For' (in Blue) and 'Goals Against' (in Red) for Premier League teams. At a glance, we can observe that teams generally have a tighter distribution for 'Goals For', suggesting a more consistent offensive performance across teams. In contrast, the 'Goals Against' shows a slightly broader spread, indicating variability in defensive performances. The box plot serves as a clear visual tool, highlighting these differences and offering insights into team strategies and areas of strength or vulnerability.")

    # Bar Chart: Average Points per Team over all Seasons
    st.write("Bar Chart: Average Points per Team over all Seasons")
    fig, ax = plt.subplots(figsize=(14, 8))
    average_points = epl_standings.groupby('Team')['Pts'].mean().sort_values(ascending=False)
    sns.barplot(x=average_points.index, y=average_points.values, palette='viridis', ax=ax)
    ax.set_title('Bar Chart: Average Points per Team over all Seasons')
    ax.set_xticklabels(average_points.index, rotation=90)
    st.pyplot(fig)
    st.write("For the overarching goal of predicting end-of-season positions, understanding historical performance is crucial. A team's consistency or inconsistency across seasons can be a significant predictor for future performance. For instance, teams that consistently score high average points might be better equipped, both in terms of skill and strategy, to secure top positions in subsequent seasons. Conversely, teams with a lower average might be more susceptible to relegation threats. Moreover, by identifying teams with similar historical performance levels, we can potentially cluster them together in later analyses, addressing one of our primary research questions. This visualization, therefore, sets the stage for deeper predictive and clustering analyses, making it highly relevant to our project's objectives.")

    # Scatterplot: Relationship between Goals For and Points
    st.write("Scatterplot: Relationship between Goals For and Points")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=epl_standings, x='GF', y='Pts', hue='Team', palette='viridis', alpha=0.7, ax=ax)
    ax.set_title('Scatterplot: Relationship between Goals For and Points')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2)
    st.pyplot(fig)
    st.write("The scatterplot confirms that goal-scoring capability is a significant determinant in a team's success, making it a potential primary feature in my predictive models. Additionally, the color-coded teams help identify outliers or unique patterns specific to certain teams. For example, if a team scores a high number of goals but still has fewer points, it might indicate issues in their defense, which they concede just as many, if not more. Such insights can further aid in feature engineering and refining my predictive algorithms.")

    # Pairplot: Pairwise relationships for select columns in EPL Standings
    st.write("Pairplot: Pairwise Relationships in EPL Standings")
    g = sns.pairplot(epl_standings[['GF', 'GA', 'Pts']])
    st.pyplot(g.fig)
    st.write("The pairplot shows how different team performance measures relate to each other. Looking at these relationships together lets you spot trends more easily. For example, while scoring more often means more points, the plot also shows that letting in goals can lower a team's points. The comparison of 'Goals For' and 'Goals Against' highlights the importance of teams being strong both in attacking and defending.")

    # Heatmap: Correlation between features in EPL Standings dataset
    st.write("Heatmap: Correlation between features in EPL Standings dataset")
    corr_matrix = epl_standings.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)
    st.write("The heatmap offers a clear insight into the interplay between different performance metrics of Premier League teams. While the correlation between 'Wins' (W) and total points underscores the primary importance of securing victories, other metrics, including 'Goals For' (GF) and 'Goals Against' (GA), also exhibit significant correlations. This tells us that when I make predictions later, I should look at many factors, not just one or two. By using all this info, my predictions can be more accurate.")

# Supportive Dataset
elif section == 'Supportive Dataset':
    st.title("Supportive Dataset: Player Transfers")
    st.write("This section provides insights based on the player transfers dataset, showcasing its relevance in supporting the primary dataset.")

    # Load the preprocessed dataset
    merged_data = pd.read_excel('/Users/aleksbuha/Desktop/BIExam/data/epl_transfers_merged.xlsx')

    # Scatter plot
    st.write("Relationship between Transfer Expenditure and League Position (2010-2022)")
    fig, ax = plt.subplots(figsize=(15,10))
    sns.scatterplot(data=merged_data, x='fee_numeric', y='Pos', hue='Team', size='Pts', sizes=(20,200), alpha=0.7, ax=ax)
    ax.invert_yaxis()
    ax.set_title('Relationship between Transfer Expenditure and League Position (2010-2022)')
    ax.set_xlabel('Transfer Expenditure (in million £)')
    ax.set_ylabel('League Position')
    st.pyplot(fig)

    # Calculate and print the correlation
    correlation = merged_data['fee_numeric'].corr(merged_data['Pos'])
    st.write(f"Correlation between Transfer Expenditure and League Position (2010-2022 Seasons): {correlation:.2f}")
    
    st.subheader("Conclusion on Scatterplot")
    st.write("There's a moderate negative correlation of −0.42 between transfer expenditure and league position. This suggests that teams which tend to invest more in player transfers often achieve better league positions. The higher the investment, the more likely a team is to secure a top spot in the EPL standings, underscoring the importance of strategic investments in players.")

    # Load the data for histogram
    merged_data = pd.read_excel('/Users/aleksbuha/Desktop/BIExam/data/epl_transfers_merged.xlsx')

    # Filter out entries where transfer fees are 0 or missing
    filtered_data = merged_data[merged_data['fee_numeric'] > 0]

    # Plotting the histogram
    st.write("Distribution of Transfer Expenditures (2010-2022)")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(data=filtered_data, x='fee_numeric', kde=True, bins=30, ax=ax)
    ax.set_title('Distribution of Transfer Expenditures (2010-2022)')
    ax.set_xlabel('Transfer Expenditure (in million £)')
    ax.set_ylabel('Number of Teams')
    st.pyplot(fig)
    
    st.subheader("Conclusion on Histogram")
    st.write("The histogram illustrates the distribution of team transfer expenditures over the 2010-2022 seasons. It provides context to the league's financial landscape, revealing that while a few teams might invest heavily in transfers, a majority operate with more modest budgets. This supports the earlier scatterplot analysis by offering a broader view of the spending behaviors of Premier League teams. It adds credibility to your main dataset by highlighting the varied financial capabilities and strategies of teams, which inevitably impact their performance and league standings.")

    # Conclusion
    st.subheader("Conclusion on the Supportive Dataset (Player Transfers)")
    st.write("The player transfers dataset not only backs up the findings from the main EPL standings dataset but also adds more depth to it, giving a detailed look at the league's workings. This extra data strengthens the conclusions drawn from the main dataset, making my analysis clearer and more trustworthy.")

# Multiple Linear Regression
elif section == 'Multiple Linear Regression':
    st.title("Multiple Linear Regression")
    st.write("Predict the end-of-season position based on specific metrics.")
    
    # Linear Regression Model
    X = epl_standings[['W', 'L', 'GF', 'GA']]
    y = epl_standings['Pos']
    linear_regressor = LinearRegression().fit(X, y)
    
    # User input fields for regression model
    w = st.number_input("Wins (W)", value=0, min_value=0, max_value=38, key='reg1')
    l = st.number_input("Losses (L)", value=0, min_value=0, max_value=38, key='reg2')
    gf = st.number_input("Goal For (GF)", value=0, key='reg3')
    ga = st.number_input("Goals Against (GA)", value=0, min_value=0, key='reg4')
    
    user_data_reg = np.array([[w, l, gf, ga]])
    predicted_position = linear_regressor.predict(user_data_reg)

    if st.button("Predict Position"):
        st.write(f"Predicted End-of-Season Position: {int(predicted_position[0])}")

# Classifications
elif section == 'Classifications':
    st.title("Classification Models")
    
    # Creating the new categorical target variable based on position bins
    conditions = [
        (epl_standings['Pos'] <= 4),
        (epl_standings['Pos'] > 4) & (epl_standings['Pos'] <= 7),
        (epl_standings['Pos'] > 7) & (epl_standings['Pos'] <= 17),
        (epl_standings['Pos'] > 17)
    ]
    choices = ['Champions League', 'Europa League', 'Other', 'Relegation Battle']
    epl_standings['Performance'] = np.select(conditions, choices)

    # Using only GD and Total Injuries as predictors
    X = epl_standings[['GD', 'Total Injuries']]
    y = epl_standings['Performance']

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the dataset into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

    # Training the Decision Tree model
    dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=123)
    dt_classifier.fit(X_train, y_train)

    # Predicting and evaluating the Decision Tree model
    y_pred_dt = dt_classifier.predict(X_test)
    
    st.subheader("Decision Tree - Classification Report:")
    st.text(classification_report(y_test, y_pred_dt))

    # Training the Naive Bayes model
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)

    # Predicting and evaluating the Naive Bayes model
    y_pred_nb = nb_classifier.predict(X_test)
    
    st.subheader("\nNaive Bayes - Classification Report:")
    st.text(classification_report(y_test, y_pred_nb))
    
    st.write("Conclusion of Decision Tree and Naive Bayes models")
    st.write("My classification models effectively sort teams into performance categories. While the Decision Tree classifier achieves an impressive 80% accuracy, the Naive Bayes model slightly edges it out with an 84% accuracy rate. Both models provide valuable insights, with the capability to identify teams' potential paths, whether they're aiming for prestigious spots like the 'Champions League' or contending for places in the 'Europa League.'")

    st.title('Predict Team Performance Category')

    # Input fields for the user
    GD_input = st.number_input('Goal Difference (GD)', min_value=-100, max_value=100, value=0)
    injuries_input = st.number_input('Total Injuries', min_value=0, max_value=50, value=0)

    # Convert the inputs to a numpy array
    user_data_class = np.array([[GD_input, injuries_input]])

    # Predicting using Decision Tree
    predicted_performance = dt_classifier.predict(user_data_class)

    if st.button('Predict Performance Category'):
        st.write(f'Predicted Performance Category: {predicted_performance[0]}')
    
# K-means
elif section == 'K-means':
    st.title('K-means Clustering on EPL Teams')
    
    # Selecting relevant features
    features = ['GD', 'Pts', 'W', 'L', 'GF', 'GA']
    data = epl_standings[features]

    # Scaling the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Elbow Method (using distortion)
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(scaled_data)
        distortions.append(kmeanModel.inertia_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K, distortions, 'bx-')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Distortion')
    ax.set_title('The Elbow Method showing the optimal k')
    st.pyplot(fig)

    st.subheader("Silhouette Scores for Various Cluster Sizes")
    scores = []
    K = range(2, 10)
    for k in K:
        model = KMeans(n_clusters=k).fit(scaled_data)
        model.fit(scaled_data)
        score = metrics.silhouette_score(scaled_data, model.labels_, metric='euclidean', sample_size=len(scaled_data))
        st.write(f"\nNumber of clusters = {k}")
        st.write(f"Silhouette score = {score:.2f}")
        scores.append(score)

    optimal_clusters = 3

    # K-means Model
    kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=123).fit(scaled_data)
    clusters = kmeans.fit_predict(scaled_data)

    st.subheader("K-means Clustering Visualization")
    # Scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=clusters, palette='viridis', s=100, alpha=0.7, ax=ax)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
    ax.set_title('Clusters')
    st.pyplot(fig)

    st.subheader('Conclusion of K-means')
    st.write("Through K-means clustering, I have unveiled intricate performance patterns among Premier League teams. By considering pivotal metrics and employing methods like the Elbow Method and silhouette analysis, my model groups teams based on nuanced performance traits, offering a deeper understanding beyond just league standings.")

    st.title("Predictions using K-means Clustering")

    # Taking input from the user
    GD = st.number_input("Goal Difference (GD)", value=0)
    Pts = st.number_input("Points (Pts)", value=0)
    W = st.number_input("Wins (W)", value=0, min_value=0, max_value=38)
    L = st.number_input("Losses (L)", value=0, min_value=0, max_value=38)
    GF = st.number_input("Goals For (GF)", value=0)
    GA = st.number_input("Goals Against (GA)", value=0)

    # Scale the user input
    user_data = np.array([[GD, Pts, W, L, GF, GA]])
    user_data_scaled = scaler.transform(user_data)

    # Predict the cluster for the user input
    user_cluster = kmeans.predict(user_data_scaled)

    # Check if the "Predict Cluster" button is pressed
    if st.button("Predict Cluster"):
        st.session_state.predict_cluster_pressed = True

    # Provide feedback to the user based on the session state
    if st.session_state.predict_cluster_pressed:
        st.write(f"The input data belongs to Cluster: {user_cluster[0]}")
    
    if user_cluster[0] == 0:
        st.write("This cluster typically represents top-performing teams.")
    elif user_cluster[0] == 1:
        st.write("This cluster typically represents mid-tier teams.")
    elif user_cluster[0] == 2:
        st.write("This cluster typically represents teams struggling or in danger of relegation.")

# Geo Map
elif section == 'Geo Map':
    st.title("Geographical Visualization of EPL Teams")
    
    st.write("""
    To visualize the performance of English Premier League teams geographically, I employed K-means clustering based on several performance metrics such as points, wins, losses, goal differences, and more. The objective was to categorize teams into distinct performance tiers over the years and then represent these clusters on a map using the location of their home stadiums.

    The clustering resulted in three clear groups, which can be visually identified on the map with distinct colors:

    - Blue (Cluster 0): Represents the Top-performing teams. These are the crème de la crème of the league, consistently vying for the title or securing spots in European competitions. Their performance metrics consistently stand out, showcasing the highest points, numerous wins, and positive goal differences.

    - Green (Cluster 1): Represents Mid-tier teams. These teams find themselves in the middle of the table more often than not. Their balanced performance metrics indicate that while they might not always be challenging for the title, they also steer clear of the regular relegation threats.

    - Red (Cluster 2): Represents the teams that are Struggling or in danger. These teams often find themselves at the bottom of the table, facing the highest losses and frequently battling to avoid relegation. Their negative goal differences further underscore the challenges they face.

    The map visualization provides an insightful geographical representation of team performances in the English Premier League. It offers a quick glance at the distribution of top-performing, mid-tier, and struggling teams based on their home stadium locations.
    """)

    # Load the data
    stadium_data = pd.read_excel("/Users/aleksbuha/Desktop/BIExam/data/GPSData.xlsx")

    # Selecting relevant features
    features = ['GD', 'Pts', 'W', 'L', 'GF', 'GA']
    data = epl_standings[features]

    # Scaling the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Performing K-means clustering
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=123).fit(scaled_data)
    clusters = kmeans.fit_predict(scaled_data)

    # Teams used for clustering
    teams_used_for_clustering = epl_standings['Team'].iloc[:len(clusters)]

    # Creating a dataframe for the teams used for clustering and the clusters
    team_clusters = pd.DataFrame({
        'Team': teams_used_for_clustering,
        'Cluster': clusters
    })

    # Merging clusters with epl_standings
    epl_standings = epl_standings.merge(team_clusters, on='Team', how='left')

    # Merging this with the stadium data for the geo-coordinates
    final_data = epl_standings.merge(stadium_data, left_on='Team', right_on='Club', how='left')

    # Plotting the map
    m = folium.Map(location=[52.3555, -1.1743], zoom_start=6)  # Centered around England

    # Define a color map for clusters
    cluster_colors = ['blue', 'green', 'red', 'grey']  # Added 'grey' for teams without cluster

    # Adding markers for each team
    for idx, row in final_data.iterrows():
        # Use grey color if Cluster is NaN, else use the cluster color
        color = cluster_colors[int(row['Cluster'])] if not pd.isna(row['Cluster']) else 'grey'
        
        folium.Marker(
            location=[row['Latitude_dd'], row['Longitude_dd']],
            popup=row['Team'],
            icon=folium.Icon(color=color)
        ).add_to(m)

    # Display the map using folium_static
    folium_static(m)

# Summary
elif section == 'Summary':
    st.title("Summary")
    st.write("A comprehensive overview and conclusion of the analyses performed in this project.")
    st.write("""
    This project presented an in-depth analysis of the English Premier League from 2000 to 2022. Using the EPL standings dataset as the primary source, key insights were derived regarding team performances, patterns, and influential factors. The integration of the player transfers dataset as a supportive dataset provided a broader perspective, shedding light on the potential influence of financial power in determining league outcomes.
    """)
    
    # Load and display an image
    image_path = "/Users/aleksbuha/Desktop/BIExam/Images/summary.png"
    st.image(image_path, use_column_width=True)
