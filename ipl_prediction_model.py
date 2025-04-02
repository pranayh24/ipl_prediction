import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, mean_absolute_error
import joblib
import warnings

warnings.filterwarnings('ignore')

# Set display options for pandas
pd.set_option('display.max_columns', None)


# Load the cricket data
def load_data(csv_path):
    print(f"Loading IPL data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} ball-by-ball records.")
    return df


# Feature engineering function for IPL-specific data
def engineer_features(df):
    print("Engineering IPL-specific features...")

    # Convert date to datetime
    df['match_date'] = pd.to_datetime(df['match_date'])

    # Extract time-based features
    df['year'] = df['match_date'].dt.year
    df['month'] = df['match_date'].dt.month
    df['is_playoff'] = df['match_date'].dt.month >= 5  # May onwards usually has playoffs

    # Create a feature for total wickets taken by each team
    df['team1_has_wickets'] = df['team1_wickets'] > 0
    df['team2_has_wickets'] = df['team2_wickets'] > 0

    # Convert over.ball format to decimal overs
    df['decimal_overs'] = df['over'].astype(float) + (df['ball'] - 1) / 6

    # Create percentage of innings completed feature
    max_overs = 20  # IPL is T20 format
    df['innings_pct_complete'] = df['decimal_overs'] / max_overs

    # For 2nd innings, create features related to chase
    df['is_second_innings'] = df['innings'] % 2 == 0

    # Create wickets_in_hand (10 - current_wickets)
    df['wickets_in_hand'] = 10 - df['current_innings_wickets']

    # Running run rate
    df['current_runrate'] = df['current_innings_runrate']

    # Calculate balls remaining in the innings
    df['balls_consumed'] = df['decimal_overs'] * 6
    df['balls_in_innings'] = np.where(df['is_second_innings'],
                                      df['balls_remaining'],
                                      max_overs * 6)
    df['balls_remaining'] = df['balls_in_innings'] - df['balls_consumed']

    # Team batting first or second
    df['batting_first'] = (df['batting_team'] == df['team1']) & (~df['is_second_innings']) | \
                          (df['batting_team'] == df['team2']) & (~df['is_second_innings'])

    # Toss winner is batting
    df['toss_winner_batting'] = df['batting_team'] == df['toss_winner']

    # Teams that won toss and chose to bat
    df['chose_to_bat'] = (df['toss_winner'] == df['batting_team']) & (df['toss_decision'] == 'bat')

    # Create target related features for 2nd innings
    df['runs_per_ball_needed'] = np.where(df['is_second_innings'] & (df['balls_remaining'] > 0),
                                          df['runs_required'] / df['balls_remaining'],
                                          np.nan)

    # IPL-specific features

    # Create team strength metrics (based on historical data)
    # For IPL, recent form is important
    if 'season' in df.columns:
        recent_seasons = df['season'].unique()[-3:]  # Last 3 seasons
        recent_df = df[df['season'].isin(recent_seasons)]
    else:
        recent_df = df

    team_batting_avg = recent_df.groupby('batting_team')['current_innings_score'].mean().reset_index()
    team_batting_avg.columns = ['team', 'batting_strength']

    team_bowling_avg = recent_df.groupby('bowling_team')['current_innings_score'].mean().reset_index()
    team_bowling_avg.columns = ['team', 'bowling_weakness']  # Higher means worse bowling

    # Merge team strengths
    df = pd.merge(df, team_batting_avg, left_on='batting_team', right_on='team', how='left')
    df = pd.merge(df, team_bowling_avg, left_on='bowling_team', right_on='team', how='left')

    # Drop unnecessary columns
    df = df.drop(['team_x', 'team_y'], axis=1, errors='ignore')

    # Calculate batting_power as the combination of batting strength and opponent's bowling weakness
    df['batting_power'] = df['batting_strength'] + df['bowling_weakness']

    # IPL Venue winning statistics
    venue_stats = df.groupby('venue')['match_winner'].agg(
        lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan).reset_index()
    venue_stats.columns = ['venue', 'most_winning_team']
    df = pd.merge(df, venue_stats, on='venue', how='left')

    # Current team has venue advantage
    df['batting_team_venue_advantage'] = df['batting_team'] == df['most_winning_team']
    df['bowling_team_venue_advantage'] = df['bowling_team'] == df['most_winning_team']

    # Wicket rate for the current innings
    df['wicket_rate'] = np.where(df['balls_consumed'] > 0,
                                 df['current_innings_wickets'] / df['balls_consumed'],
                                 0)

    # Momentum features - runs in last few overs
    # Create a match-innings identifier
    df['match_innings_id'] = df['match_id'] + '_' + df['innings'].astype(str)

    # Sort by match, innings, and ball
    df = df.sort_values(['match_innings_id', 'ball_number'])

    # Calculate runs in last 6 balls and last 12 balls
    df['runs_last_6_balls'] = df.groupby('match_innings_id')['runs_total'].rolling(window=6,
                                                                                   min_periods=1).sum().reset_index(
        drop=True)
    df['runs_last_12_balls'] = df.groupby('match_innings_id')['runs_total'].rolling(window=12,
                                                                                    min_periods=1).sum().reset_index(
        drop=True)

    # Calculate wickets in last 6 balls and last 12 balls
    df['wickets_last_6_balls'] = df.groupby('match_innings_id')['wicket'].rolling(window=6,
                                                                                  min_periods=1).sum().reset_index(
        drop=True)
    df['wickets_last_12_balls'] = df.groupby('match_innings_id')['wicket'].rolling(window=12,
                                                                                   min_periods=1).sum().reset_index(
        drop=True)

    # IPL-specific powerplay and death overs features
    df['is_powerplay'] = df['decimal_overs'] <= 6.0
    df['is_middle_overs'] = (df['decimal_overs'] > 6.0) & (df['decimal_overs'] <= 15.0)
    df['is_death_overs'] = df['decimal_overs'] > 15.0

    # Filter only the last ball of each over to create a dataset for estimating final score
    df_by_over = df.groupby(['match_id', 'innings', 'over']).last().reset_index()

    # Get the final score for each innings to use as target variable
    final_scores = df.groupby(['match_id', 'innings'])['current_innings_score'].max().reset_index()
    final_scores.columns = ['match_id', 'innings', 'final_score']

    # For match winner prediction, we need the complete match data
    match_results = df.groupby('match_id')[['team1', 'team2', 'match_winner']].first().reset_index()

    # Merge final scores back to the by-over data
    df_by_over = pd.merge(df_by_over, final_scores, on=['match_id', 'innings'], how='left')

    # Create remaining_score feature (how many more runs were scored from this point)
    df_by_over['remaining_score'] = df_by_over['final_score'] - df_by_over['current_innings_score']

    print("Feature engineering complete.")
    return df, df_by_over, match_results


# Create datasets for score prediction and win prediction
def prepare_score_prediction_data(df_by_over):
    print("Preparing IPL score prediction dataset...")

    # Define features and target for score prediction
    features = [
        'batting_team', 'bowling_team', 'venue', 'innings',
        'current_innings_score', 'current_innings_wickets', 'decimal_overs',
        'innings_pct_complete', 'current_runrate', 'wickets_in_hand',
        'batting_strength', 'bowling_weakness', 'batting_power',
        'batting_team_venue_advantage', 'wicket_rate',
        'runs_last_6_balls', 'runs_last_12_balls',
        'wickets_last_6_balls', 'wickets_last_12_balls',
        'is_second_innings', 'is_powerplay', 'is_middle_overs', 'is_death_overs',
        'year'
    ]

    # Filter rows with at least 5 overs played to make meaningful predictions
    score_df = df_by_over[df_by_over['decimal_overs'] >= 5.0].copy()

    # Handle missing values
    for col in features:
        if col in score_df.columns and score_df[col].dtype in [np.float64, np.int64]:
            score_df[col] = score_df[col].fillna(0)

    # Features and target
    X_score = score_df[features].copy()
    y_score = score_df['final_score'].copy()

    # Split data
    X_train_score, X_test_score, y_train_score, y_test_score = train_test_split(
        X_score, y_score, test_size=0.2, random_state=42
    )

    print(f"IPL score prediction dataset: {len(X_train_score)} training samples, {len(X_test_score)} test samples")
    return X_train_score, X_test_score, y_train_score, y_test_score, features


def prepare_win_prediction_data(df):
    print("Preparing IPL win prediction dataset...")

    # Get a subset of data (one record per over per match)
    win_df = df.groupby(['match_id', 'innings', 'over']).last().reset_index()

    # Filter to include only matches with a declared winner
    win_df = win_df[win_df['match_winner'].notna() & (win_df['match_winner'] != '')].copy()

    # Create target variable: is_batting_team_winner
    win_df['is_batting_team_winner'] = win_df['batting_team'] == win_df['match_winner']

    # Define features for win prediction
    win_features = [
        'batting_team', 'bowling_team', 'venue', 'innings',
        'current_innings_score', 'current_innings_wickets', 'decimal_overs',
        'team1_score', 'team1_wickets', 'team1_runrate',
        'team2_score', 'team2_wickets', 'team2_runrate',
        'is_second_innings', 'toss_winner_batting', 'chose_to_bat',
        'batting_strength', 'bowling_weakness',
        'batting_team_venue_advantage', 'bowling_team_venue_advantage',
        'is_powerplay', 'is_middle_overs', 'is_death_overs',
        'is_playoff'  # IPL-specific playoff feature
    ]

    # Add chase-specific features for second innings
    win_df_2nd = win_df[win_df['is_second_innings']].copy()
    if not win_df_2nd.empty:
        win_df_2nd = win_df_2nd.dropna(subset=['target', 'runs_required', 'balls_remaining'])

        # Replace zero balls_remaining with a small value to avoid division by zero
        win_df_2nd['balls_remaining_safe'] = win_df_2nd['balls_remaining'].replace(0, 0.1)

        # Calculate required run rate safely (cap at a reasonable maximum)
        win_df_2nd['chase_required_rr'] = (win_df_2nd['runs_required'] * 6 /
                                           win_df_2nd['balls_remaining_safe'])
        # Cap the required run rate to a reasonable maximum value (e.g., 36 = 6 runs per ball)
        win_df_2nd['chase_required_rr'] = win_df_2nd['chase_required_rr'].clip(upper=36.0)

        win_df_2nd['current_rr'] = win_df_2nd['current_runrate']
        win_df_2nd['rr_difference'] = win_df_2nd['current_rr'] - win_df_2nd['chase_required_rr']

        # Merge back with main dataframe
        win_df = pd.concat([
            win_df[~win_df['is_second_innings']],
            win_df_2nd
        ])

    # Add second innings features to the feature list
    if 'chase_required_rr' in win_df.columns:
        second_innings_features = ['target', 'runs_required', 'balls_remaining',
                                   'chase_required_rr', 'current_rr', 'rr_difference']
        win_features.extend([f for f in second_innings_features if f in win_df.columns])

    # Handle missing values
    for col in win_features:
        if col in win_df.columns and win_df[col].dtype in [np.float64, np.int64]:
            win_df[col] = win_df[col].fillna(0)

    # Make sure there are no infinity values
    for col in win_features:
        if col in win_df.columns and win_df[col].dtype in [np.float64, np.int64]:
            win_df[col] = win_df[col].replace([np.inf, -np.inf], np.nan)
            win_df[col] = win_df[col].fillna(win_df[col].mean() if win_df[col].mean() == win_df[col].mean() else 0)

    # Features and target
    X_win = win_df[win_features].copy()
    y_win = win_df['is_batting_team_winner'].copy()

    # Split data
    X_train_win, X_test_win, y_train_win, y_test_win = train_test_split(
        X_win, y_win, test_size=0.2, random_state=42, stratify=y_win
    )

    print(f"IPL win prediction dataset: {len(X_train_win)} training samples, {len(X_test_win)} test samples")
    return X_train_win, X_test_win, y_train_win, y_test_win, win_features


# Build the score prediction model for IPL
def build_score_prediction_model(X_train, X_test, y_train, y_test, features):
    print("Building IPL score prediction model...")

    # Identify categorical and numerical features
    categorical_features = [col for col in features
                            if X_train[col].dtype == 'object' or X_train[col].nunique() < 10]
    numerical_features = [col for col in features if col not in categorical_features]

    # Create preprocessing steps
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ]
    )

    # Create models - using IPL-optimized parameters
    models = {
        'xgboost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                       random_state=42, n_jobs=-1))
        ]),
        'gradient_boosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(n_estimators=250, max_depth=6, random_state=42))
        ]),
        'random_forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1))
        ])
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {
            'model': model,
            'mae': mae,
            'rmse': rmse
        }

        print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Find best model
    best_model_name = min(results, key=lambda x: results[x]['mae'])
    print(f"Best IPL score prediction model: {best_model_name} with MAE: {results[best_model_name]['mae']:.2f}")

    # Return best model
    return results[best_model_name]['model']


# Build the win prediction model for IPL
def build_win_prediction_model(X_train, X_test, y_train, y_test, features):
    print("Building IPL win prediction model...")

    # Identify categorical and numerical features
    categorical_features = [col for col in features
                            if X_train[col].dtype == 'object' or X_train[col].nunique() < 10]
    numerical_features = [col for col in features if col not in categorical_features]

    # Create preprocessing steps
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ]
    )

    # Create models - using IPL-optimized parameters
    models = {
        'xgboost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                        random_state=42, n_jobs=-1))
        ]),
        'gradient_boosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', GradientBoostingClassifier(n_estimators=250, max_depth=6, random_state=42))
        ]),
        'random_forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1))
        ])
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class

        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': classification_report(y_test, y_pred)
        }

        print(f"{name} - Accuracy: {accuracy:.4f}")
        print(results[name]['report'])

    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    print(f"Best IPL win prediction model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")

    # Return best model
    return results[best_model_name]['model']


# Save the models
def save_models(score_model, win_model):
    print("Saving IPL prediction models...")
    joblib.dump(score_model, 'ipl_score_prediction_model.joblib')
    joblib.dump(win_model, 'ipl_win_prediction_model.joblib')
    print("Models saved.")


# Main function to train both models
def train_models(csv_path):
    # Load data
    df = load_data(csv_path)

    # Engineer features
    df, df_by_over, match_results = engineer_features(df)

    # Prepare data for score prediction
    X_train_score, X_test_score, y_train_score, y_test_score, score_features = prepare_score_prediction_data(df_by_over)

    # Build score prediction model
    score_model = build_score_prediction_model(X_train_score, X_test_score, y_train_score, y_test_score, score_features)

    # Prepare data for win prediction
    X_train_win, X_test_win, y_train_win, y_test_win, win_features = prepare_win_prediction_data(df)

    # Build win prediction model
    win_model = build_win_prediction_model(X_train_win, X_test_win, y_train_win, y_test_win, win_features)

    # Save models
    save_models(score_model, win_model)

    return score_model, win_model, score_features, win_features


if __name__ == "__main__":
    # Train the models
    csv_path = input("Enter the path to the IPL ball-by-ball CSV file: ")
    score_model, win_model, score_features, win_features = train_models(csv_path)