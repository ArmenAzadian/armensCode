from nba_api.stats.endpoints import leaguestandings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def fetch_team_data():
    standings = leaguestandings.LeagueStandings().get_data_frames()[0]
    data = {
        'team': standings['TeamName'],
        'points_per_game': standings['PointsPG'],
        'opp_points_per_game': standings['OppPointsPG'],
        'point_differential': standings['DiffPointsPG'],
        'lead_in_rebounds': standings['LeadInReb'].apply(lambda x: int(x.split('-')[0]) if isinstance(x, str) else x),
        'fewer_turnovers': standings['FewerTurnovers'].apply(lambda x: int(x.split('-')[0]) if isinstance(x, str) else x),
        'score_100_plus': standings['Score100PTS'].apply(lambda x: int(x.split('-')[0]) if isinstance(x, str) else x),
        'opp_score_100_plus': standings['OppScore100PTS'].apply(lambda x: int(x.split('-')[0]) if isinstance(x, str) else x),
    }
    df = pd.DataFrame(data)
    df['playoff'] = df['point_differential'].rank(ascending=False).apply(lambda x: 1 if x <= 16 else 0)
    return df

team_name = input("Enter the name of the team: ").strip()

data = fetch_team_data()

if team_name not in data['team'].values:
    print(f"The team '{team_name}' was not found in the dataset.")
else:
    index = [
        'points_per_game',
        'opp_points_per_game',
        'point_differential',
        'lead_in_rebounds',
        'fewer_turnovers',
        'score_100_plus',
        'opp_score_100_plus'
    ]
    target = 'playoff'

    X = data[index]
    y = data[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"\nModel Accuracy: {accuracy:.2f}")

    team_data_scaled = scaler.transform(data[data['team'] == team_name][index])

    prediction = model.predict(team_data_scaled)[0]

    coefficients = pd.DataFrame(model.coef_.T, index, columns=['Coefficient'])
    print("\nCoefficients:")
    print("The features with the highest absolute coefficients are the most influential in predicting playoff appearances.")
    print("Positive coefficients indicate a positive relationship with GPA, while negative coefficients indicate a negative relationship.")
    print(coefficients)

    print(f"\nTeam: {team_name}")
    if prediction == 1:
        print(f"The {team_name} are predicted to qualify for the playoffs.")
    else:
        print(f"The {team_name} are predicted to not qualify for the playoffs.")
