import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
world_cup = pd.read_csv("World_cup_2023.csv")
results = pd.read_csv("results.csv")

# Sidebar navigation
main_nav_options = ["Team Titles", "Team Win Percentage", "Team Matches Won", "Recent ODI Ranking", "Team Wins in World Cup", "India Wins Against Other Teams", "India Win Percentage Against Each Team", "Predictions"]

# Secondary navigation options for "Predictions"
predictions_options = ["Qualified Teams for Semi-finals","ML Model Accuracy","Fixture Prediction", "Predict World-Cup Winner"]

# Main navigation radio button
nav = st.sidebar.radio("Analysis", main_nav_options)


st.title("Cricket Analysis and Prediction")

if nav == "Team Titles":
    st.header("Number of Titles Won by Each Team")
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Team_name', y='Titles', data=world_cup)
    plt.xticks(rotation=90, ha='center')  # Rotate team names vertically
    st.pyplot(plt)

elif nav == "Team Win Percentage":
    st.header("Win Percentage in ODI by Each Team")
    
    # Set the figure size and plot the bar chart
    plt.figure(figsize=(20, 5))
    sns.barplot(x='Team_name', y='Win_percentage_ODI', data=world_cup)
    
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=90, ha='center')
    
    # Display the bar chart using Streamlit
    st.pyplot(plt)

    # Set up the pie chart
    st.header("Win Percentage in ODI by Each Team (Pie Chart)")

    # Create a pie chart using the same data and aggregation
    pie_data = world_cup.groupby('Team_name')['Win_percentage_ODI'].sum().reset_index()

    # Set the figure size for the pie chart
    plt.figure(figsize=(10, 10))

    # Plot the pie chart
    plt.pie(pie_data['Win_percentage_ODI'], labels=pie_data['Team_name'], autopct='%1.1f%%', startangle=140)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    # Display the pie chart using Streamlit
    st.pyplot(plt)

elif nav == "Team Matches Won":
    st.header("Number of Matches Won in World Cup by Each Team")
    sns.barplot(x='Team_name', y='WC_match_won', data=world_cup)
    plt.xticks(rotation=90, ha='center')
    st.pyplot(plt)

elif nav == "Recent ODI Ranking":
    st.header("Recent ODI Ranking")
    sns.barplot(x='Team_name', y='Rating', data=world_cup)
    plt.xticks(rotation=90, ha='center')
    st.pyplot(plt)

elif nav == "Team Wins in World Cup":
    st.header("Team Wins in ODI World Cup 2023")
    # Add your code here
    team_wins_wc_2023 = results.groupby("Team_1")["Winner"].count().reset_index()
    sns.barplot(x='Team_1', y='Winner', data=team_wins_wc_2023)
    plt.xticks(rotation=90, ha='center')
    st.pyplot(plt)


elif nav == "India Wins Against Other Teams":
    st.header("India Wins in ODIs Against Other Teams")
    # Add your code here
    df = results[(results["Team_1"] == 'India') | (results["Team_2"] == 'India')]
    India = df.iloc[:]
    India.head()

    India_win = India[India['Winner'] == 'India']
    st.dataframe(India_win)

    # No of wins in ODIs against other teams - Bar plot
    exclude = 'India'
    filtered_data_team1 = India_win[India_win['Team_1'] != exclude]
    st.subheader("No of wins in ODIs against other teams (Team 1)")
    st.bar_chart(filtered_data_team1['Team_1'].value_counts().head(5))

    exclude = 'India'
    filtered_data_team2 = India_win[India_win['Team_2'] != exclude]
    st.subheader("No of wins in ODIs against other teams (Team 2)")
    st.bar_chart(filtered_data_team2['Team_2'].value_counts().head(5))


elif nav == "India Win Percentage Against Each Team":
   # Load the data from the CSV file
    world_cup = pd.read_csv("results.csv")

    # Filter rows where India is either Team_1 or Team_2
    india_matches = world_cup[(world_cup["Team_1"] == "India") | (world_cup["Team_2"] == "India")]

    # Create a dictionary to store win counts against each team
    team_win_counts = {}

    # Iterate over each row and update the win counts
    for _, match in india_matches.iterrows():
        opponent = match["Team_1"] if match["Team_2"] == "India" else match["Team_2"]
        if opponent not in team_win_counts:
            team_win_counts[opponent] = 0
        if match["Winner"] == "India":
            team_win_counts[opponent] += 1

    # Calculate total matches and win percentages
    total_matches = sum(team_win_counts.values())
    win_percentages = {team: (wins / total_matches) * 100 for team, wins in team_win_counts.items()}

    # Create a pie chart
    st.header("Win Percentage of India Against Each Team")
    fig, ax = plt.subplots(figsize=(5, 5))
    threshold = 1  # Set a threshold for slice size
    labels = [f"{team} ({percentage:.1f}%)"
            if percentage > threshold else ''
            for team, percentage in win_percentages.items()]
    ax.pie(win_percentages.values(), labels=labels, autopct='', startangle=140, counterclock=False, textprops={'fontsize': 8})
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

elif nav == "Predictions":
    worldcup_teams = ['England', 'South Africa', 'West Indies',
                      'Pakistan', 'New Zealand', 'Sri Lanka', 'Afghanistan',
                      'Australia', 'Bangladesh', 'India']

    df_teams_1 = results[results['Team_1'].isin(worldcup_teams)]
    df_teams_2 = results[results['Team_2'].isin(worldcup_teams)]

    df_teams = pd.concat((df_teams_1, df_teams_2))
    df_teams.drop_duplicates(inplace=True)  # Add inplace=True to modify the DataFrame in place
    count = df_teams.shape[0]  # Assuming you want the row count

    #removing useless columns
    df_teams_2019 = df_teams.drop(['Date', 'Margin', 'Ground'], axis=1)
    #Replacing the winner column by team no instead of name
    df_teams_2019 = df_teams_2019.reset_index(drop=True)
    df_teams_2019.loc[df_teams_2019.Winner == df_teams_2019.Team_1, 'winning_team'] = 1
    df_teams_2019.loc[df_teams_2019.Winner == df_teams_2019.Team_2, 'winning_team'] = 2
    df_teams_2019 = df_teams_2019.drop(['winning_team'], axis= 1)
    

     # One-hot encoding
    final = pd.get_dummies(df_teams_2019, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'], dtype=int)
    X = final.drop(['Winner'], axis=1)
    y = final["Winner"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Train the machine learning model
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
    rf.fit(X_train, y_train)


    pre_nav = st.sidebar.radio("Predictions", predictions_options)
    
    if pre_nav == "Qualified Teams for Semi-finals":
        st.header("Qualified Teams for Semi-finals")

        # Display the filtered DataFrame
        st.dataframe(df_teams)
        # Display the count of rows
        st.write(f"Number of rows: {count}")

        st.header("Filtered Teams Data")
        # Display the filtered DataFrame
        st.dataframe(df_teams)
        # Display the count of rows
        st.write(f"Number of rows: {count}")

        # Display the DataFrame after removing columns
        st.header("DataFrame After Removing useless Columns and Replacing the winner column by team no instead of name")
        st.dataframe(df_teams_2019)

    elif pre_nav == "ML Model Accuracy":
        # Using ML to predict the winner
        st.title("Machine Learning Model for Predicting Winners")
        st.header("Data Preprocessing")

        # Display the first few rows of the processed DataFrame
        st.dataframe(final.head())

        # Model evaluation
        st.header("Model Evaluation")
        train_accuracy = rf.score(X_train, y_train)
        test_accuracy = rf.score(X_test, y_test)
        st.write("Training set accuracy:", '%.3f' % train_accuracy)
        st.write("Test set accuracy:", '%.3f' % test_accuracy)

    elif pre_nav == "Fixture Prediction":
        
        # Accessing Fixtures and Icc_ranking datasets
        ranking = pd.read_csv('Icc_ranking.csv')
        fixtures = pd.read_csv('Fixtures.csv')

        # Creating a Streamlit sidebar to display raw data
        st.sidebar.subheader("Raw Data")
        st.sidebar.write("ICC Ranking Dataset:")
        st.sidebar.write(ranking)
        st.sidebar.write("Fixtures Dataset:")
        st.sidebar.write(fixtures)

        # Creating a Streamlit section to display processed data
        st.title("Fixture Predictions")
        st.header("Accessing Fixtures and Icc_ranking datasets")

        # Creating a DataFrame to store the predictions
        pred_set = []

        # Modifying fixtures DataFrame
        fixtures.insert(1, 'first_position', fixtures['Team_1'].map(ranking.set_index('Team_name')['Team_ranking']))
        fixtures.insert(2, 'second_position', fixtures['Team_2'].map(ranking.set_index('Team_name')['Team_ranking']))
        fixtures = fixtures.iloc[:45, :]
        fixtures['first_position'] = fixtures['first_position'].fillna(fixtures['first_position'].mean())
        fixtures['second_position'] = fixtures['second_position'].fillna(fixtures['second_position'].mean())

        # Looping through fixtures to create the prediction set
        for index, row in fixtures.iterrows():
            if row['first_position'] < row['second_position']:
                pred_set.append({'Team_1': row['Team_1'], 'Team_2': row['Team_2'], 'winning_team': None})
            else:
                pred_set.append({'Team_1': row['Team_2'], 'Team_2': row['Team_1'], 'winning_team': None})

        # Creating a DataFrame for the prediction set
        pred_set = pd.DataFrame(pred_set)
        back_pred_set = pred_set  # Backup of the dataset

        # One-hot encoding the prediction set
        pred_set = pd.get_dummies(pred_set, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'], dtype=int)

        # Adding missing columns to the prediction set
        missing_cols = set(final.columns) - set(pred_set.columns)
        for c in missing_cols:
            pred_set[c] = 0
        pred_set = pred_set[final.columns]
        pred_set = pred_set.drop(['Winner'], axis=1)

        # Displaying the first few rows of the processed prediction set
        st.subheader("Processed Prediction Set")
        st.dataframe(pred_set.head())

        # Making predictions using the machine learning model
        predictions = rf.predict(pred_set)

    elif pre_nav == "Predict World-Cup Winner":
        # User input for fixture prediction
        st.header("Predict World-Cup Winner")
        st.write("Enter the names of two teams to predict the winner:")
            
        # Add input fields for the user to enter team names
        team1_input = st.text_input("Enter Team 1:")
        team2_input = st.text_input("Enter Team 2:")

        # Convert team names to lowercase for case insensitivity
        team1_input_lower = team1_input.lower()
        team2_input_lower = team2_input.lower()

        # Create a button to trigger the prediction
        if st.button("Predict Winner"):
            # Check if both team names are valid
            valid_teams_lower = set(team.lower() for team in worldcup_teams)  # Use lowercase set for case insensitivity
            if team1_input_lower not in valid_teams_lower or team2_input_lower not in valid_teams_lower:
                st.error("Please enter valid team names.")
            elif team1_input_lower == team2_input_lower:
                st.error("Teams must be different.")
            else:
                # Create two prediction sets with different team orders
                prediction_set_1 = pd.DataFrame({'Team_1': [team1_input_lower], 'Team_2': [team2_input_lower]})
                prediction_set_2 = pd.DataFrame({'Team_1': [team2_input_lower], 'Team_2': [team1_input_lower]})
                
                # Ensure the same one-hot encoding is applied as during training for both sets
                prediction_set_1_encoded = pd.get_dummies(prediction_set_1, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'], dtype=int)
                prediction_set_2_encoded = pd.get_dummies(prediction_set_2, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'], dtype=int)

                # Align the columns with the training data to ensure consistency for both sets
                prediction_set_1_encoded = prediction_set_1_encoded.reindex(columns=final.columns, fill_value=0)
                prediction_set_2_encoded = prediction_set_2_encoded.reindex(columns=final.columns, fill_value=0)
                
                # Drop the 'Winner' column as it's not needed for prediction
                prediction_set_1_encoded = prediction_set_1_encoded.drop(['Winner'], axis=1, errors='ignore')
                prediction_set_2_encoded = prediction_set_2_encoded.drop(['Winner'], axis=1, errors='ignore')
                
                # Use your machine learning model to predict the winner for both sets
                predictions_1 = rf.predict(prediction_set_1_encoded)
                predictions_2 = rf.predict(prediction_set_2_encoded)
                
                # Display the predicted outcome for both sets
                st.subheader("Predicted Fixture Outcomes:")
                st.write(f"Match: {team1_input} vs {team2_input}")
                st.write("Predicted Winner", team1_input if predictions_1[0] == 1 else team2_input)
