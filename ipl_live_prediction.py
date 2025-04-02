import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')


# Load the IPL prediction models
def load_models():
    print("Loading IPL prediction models...")
    try:
        score_model = joblib.load('ipl_score_prediction_model.joblib')
        win_model = joblib.load('ipl_win_prediction_model.joblib')
        print("IPL models loaded successfully.")
        return score_model, win_model
    except Exception as e:
        print(f"Error loading IPL models: {e}")
        return None, None


# Function to predict estimated final score for IPL matches
def predict_final_score(live_data, score_model):
    """
    Predicts the final score of an IPL innings based on current match state.

    Args:
        live_data (dict): Dictionary with current match data
        score_model: Trained model for IPL score prediction

    Returns:
        int: Estimated final score
    """
    try:
        # Create a dataframe with the required features
        df = pd.DataFrame([live_data])

        # Handle overs conversion
        if 'currentOvers' in df.columns:
            try:
                overs_parts = str(df['currentOvers'].iloc[0]).split('.')
                if len(overs_parts) == 2:
                    over_part = int(overs_parts[0])
                    ball_part = int(overs_parts[1])
                    # Convert to decimal overs
                    df['decimal_overs'] = over_part + ball_part / 6
                else:
                    df['decimal_overs'] = float(df['currentOvers'].iloc[0])
            except:
                # Fallback if conversion fails
                df['decimal_overs'] = float(df['currentOvers'].iloc[0])

        # Essential match features
        df['batting_team'] = df['battingTeam']
        df['bowling_team'] = df['team2'] if df['battingTeam'] == df['team1'] else df['team1']
        df['current_innings_score'] = df['currentRuns']
        df['current_innings_wickets'] = df['currentWickets']

        # Use venue data if available
        if 'venue' not in df.columns and 'venue' in live_data:
            df['venue'] = live_data['venue']
        elif 'venue' not in df.columns:
            df['venue'] = 'unknown_venue'  # Default value

        # Determine innings
        is_second_innings = False
        if 'target' in live_data and live_data['target'] is not None:
            df['innings'] = 2
            is_second_innings = True
        else:
            df['innings'] = 1

        df['is_second_innings'] = is_second_innings

        # Calculate wickets in hand and their value
        df['wickets_in_hand'] = 10 - df['currentWickets']

        # IPL is always 20 overs
        max_overs = 20

        # Calculate percentage of innings completed
        df['innings_pct_complete'] = df['decimal_overs'] / max_overs

        # Current run rate calculation
        current_rr = 0
        if df['decimal_overs'].iloc[0] > 0:
            current_rr = df['currentRuns'].iloc[0] / df['decimal_overs'].iloc[0]
        df['current_runrate'] = current_rr

        # Add IPL-specific phase features
        df['is_powerplay'] = df['decimal_overs'] <= 6.0
        df['is_middle_overs'] = (df['decimal_overs'] > 6.0) & (df['decimal_overs'] <= 15.0)
        df['is_death_overs'] = df['decimal_overs'] > 15.0

        # Set default values for required features
        features_needed = [
            'batting_strength', 'bowling_weakness', 'batting_power',
            'batting_team_venue_advantage', 'wicket_rate',
            'runs_last_6_balls', 'runs_last_12_balls',
            'wickets_last_6_balls', 'wickets_last_12_balls',
            'year'
        ]

        for feature in features_needed:
            if feature not in df.columns:
                df[feature] = 0  # Default value

        # Wicket rate
        if df['decimal_overs'].iloc[0] > 0:
            df['wicket_rate'] = df['currentWickets'].iloc[0] / (df['decimal_overs'].iloc[0] * 6)

        # Set current year
        df['year'] = pd.Timestamp.now().year

        # Try to predict using the model
        try:
            print("Making IPL score prediction with model...")
            estimated_score = score_model.predict(df)[0]
            print(f"Model prediction: {estimated_score}")
            return max(int(estimated_score), df['currentRuns'].iloc[0])
        except Exception as e:
            print(f"Error in model prediction: {e}")
            # Fall back to IPL-specific formula
            return calculate_ipl_estimated_score(live_data)

    except Exception as e:
        print(f"Error predicting IPL score: {e}")
        # Fall back to IPL-specific formula
        return calculate_ipl_estimated_score(live_data)


def calculate_ipl_estimated_score(live_data):
    """
    IPL-specific formula for score prediction when the model fails.
    Takes into account IPL-specific scoring patterns.

    Args:
        live_data (dict): Dictionary with current match data

    Returns:
        int: Estimated final score
    """
    print("Using IPL-specific formula for score prediction...")

    # Extract needed data
    current_runs = live_data['currentRuns']
    current_wickets = live_data['currentWickets']

    # Parse overs
    try:
        overs_parts = str(live_data['currentOvers']).split('.')
        if len(overs_parts) == 2:
            completed_overs = int(overs_parts[0])
            balls_in_current_over = int(overs_parts[1])
        else:
            completed_overs = int(float(live_data['currentOvers']))
            balls_in_current_over = 0
    except:
        # Default if parsing fails
        completed_overs = int(float(live_data['currentOvers']))
        balls_in_current_over = 0

    # Calculate decimal overs
    decimal_overs = completed_overs + (balls_in_current_over / 6)

    # IPL is always 20 overs
    max_overs = 20

    # Remaining overs
    remaining_overs = max_overs - decimal_overs

    # Calculate current run rate
    current_rr = 0
    if decimal_overs > 0:
        current_rr = current_runs / decimal_overs

    # IPL-specific wicket factor - wickets are more valuable in IPL
    wicket_factor = 1 - (current_wickets * 0.08)  # 8% reduction per wicket
    wicket_factor = max(0.5, wicket_factor)  # At least 50% of current rate

    # IPL-specific phase factors
    # Powerplay (overs 1-6)
    additional_runs = 0
    if decimal_overs <= 6.0:
        # If still in powerplay, expect slightly higher scoring in middle overs
        remaining_powerplay = 6.0 - decimal_overs
        remaining_middle = 9.0  # Overs 7-15
        remaining_death = 5.0  # Overs 16-20

        # Expected run rates for each phase (based on IPL averages)
        powerplay_rr = max(current_rr, 8.0)  # At least 8.0 in powerplay
        middle_rr = powerplay_rr * 0.9  # 90% of powerplay rate
        death_rr = powerplay_rr * 1.4  # 140% of powerplay rate

        # Calculate expected additional runs
        additional_runs = (remaining_powerplay * powerplay_rr) + (remaining_middle * middle_rr) + (
                    remaining_death * death_rr)

    # Middle overs (overs 7-15)
    elif decimal_overs <= 15.0:
        # If in middle overs, expect acceleration in death overs
        remaining_middle = 15.0 - decimal_overs
        remaining_death = 5.0

        # Expected run rates for each phase
        middle_rr = current_rr
        death_rr = current_rr * 1.5  # 50% increase in death overs

        # Calculate expected additional runs
        additional_runs = (remaining_middle * middle_rr) + (remaining_death * death_rr)

    # Death overs (overs 16-20)
    else:
        # In death overs, maintain or increase current rate
        # The closer to the end, the higher the rate
        overs_into_death = decimal_overs - 15.0
        proportion_of_death = overs_into_death / 5.0

        # Progressive acceleration factor
        acceleration = 1.0 + (0.5 * (1 - proportion_of_death))
        death_rr = current_rr * acceleration

        # Calculate expected additional runs
        additional_runs = remaining_overs * death_rr

    # Apply wicket factor
    adjusted_additional_runs = additional_runs * wicket_factor

    # Handle second innings chase scenarios
    if 'target' in live_data and live_data['target'] is not None:
        target = live_data['target']
        runs_required = target - current_runs

        if runs_required <= 0:
            return current_runs  # Already reached target

        # Required run rate
        required_rr = runs_required / remaining_overs if remaining_overs > 0 else float('inf')

        # IPL-specific chase adjustment - teams tend to chase more successfully in IPL
        chase_weight = min(0.8, decimal_overs / max_overs)  # Cap at 0.8

        # More weight to current RR as the innings progresses
        # IPL teams often accelerate dramatically at the end
        blended_rr = (current_rr * chase_weight) + (required_rr * (1 - chase_weight))

        # Add IPL-specific clutch factor - teams often find a way to win
        clutch_factor = 1.1 if required_rr < 12 else 0.95  # If RRR < 12, teams often exceed predictions

        # Project score with blended rate plus the clutch factor
        chase_additional_runs = min(runs_required, blended_rr * remaining_overs * clutch_factor)
        estimated_score = current_runs + chase_additional_runs

        # Don't exceed target by too much in a chase
        return min(int(estimated_score), target + 8)

    # For first innings, use the adjusted projection with venue adjustment
    estimated_score = current_runs + adjusted_additional_runs

    # Add IPL-specific venue adjustment
    venue_factor = 1.0
    if 'venue' in live_data:
        venue = live_data['venue'].lower()

        # IPL venue-specific scoring patterns
        high_scoring_venues = ['chinnaswamy', 'wankhede', 'brabourne', 'indore', 'feroz shah kotla', 'arun jaitley']
        low_scoring_venues = ['chepauk', 'eden gardens', 'rajiv gandhi', 'pune']

        if any(v in venue for v in high_scoring_venues):
            venue_factor = 1.15  # 15% boost for high-scoring IPL venues
        elif any(v in venue for v in low_scoring_venues):
            venue_factor = 0.92  # 8% reduction for low-scoring IPL venues

    estimated_score = estimated_score * venue_factor

    # Add team-specific adjustment if available
    if 'battingTeam' in live_data:
        batting_team = live_data['battingTeam'].lower()

        # IPL teams known for high/low scoring (update based on recent seasons)
        high_scoring_teams = ['mumbai indians', 'royal challengers bangalore', 'punjab kings', 'rajasthan royals']
        low_scoring_teams = ['chennai super kings', 'delhi capitals', 'sunrisers hyderabad']

        if any(team.lower() in batting_team for team in high_scoring_teams):
            estimated_score *= 1.05  # 5% boost for high-scoring teams
        elif any(team.lower() in batting_team for team in low_scoring_teams):
            estimated_score *= 0.98  # 2% reduction for low-scoring teams

    return int(estimated_score)


# Function to predict match winner probability for IPL matches
def predict_winner_probability(live_data, win_model):
    try:
        # Create a dataframe with the required features
        df = pd.DataFrame([live_data])

        # Handle overs conversion
        if 'currentOvers' in df.columns:
            try:
                overs_parts = str(df['currentOvers'].iloc[0]).split('.')
                if len(overs_parts) == 2:
                    over_part = int(overs_parts[0])
                    ball_part = int(overs_parts[1])
                    # Convert to decimal overs
                    df['decimal_overs'] = over_part + ball_part / 6
                else:
                    df['decimal_overs'] = float(df['currentOvers'].iloc[0])
            except:
                # Fallback if conversion fails
                df['decimal_overs'] = float(df['currentOvers'].iloc[0])

        # Essential match features
        df['batting_team'] = df['battingTeam']
        df['bowling_team'] = df['team2'] if df['battingTeam'] == df['team1'] else df['team1']
        df['current_innings_score'] = df['currentRuns']
        df['current_innings_wickets'] = df['currentWickets']

        # Use venue data if available
        if 'venue' not in df.columns and 'venue' in live_data:
            df['venue'] = live_data['venue']
        elif 'venue' not in df.columns:
            df['venue'] = 'unknown_venue'  # Default value

        # Determine innings
        is_second_innings = False
        if 'target' in live_data and live_data['target'] is not None:
            df['innings'] = 2
            is_second_innings = True
        else:
            df['innings'] = 1

        df['is_second_innings'] = is_second_innings

        # Toss information
        df['toss_winner_batting'] = df['battingTeam'].iloc[0] == df['tossWinner'].iloc[0]
        df['chose_to_bat'] = df['toss_winner_batting'] & (df['tossDecision'].iloc[0] == 'bat')

        # IPL-specific phase features
        df['is_powerplay'] = df['decimal_overs'] <= 6.0
        df['is_middle_overs'] = (df['decimal_overs'] > 6.0) & (df['decimal_overs'] <= 15.0)
        df['is_death_overs'] = df['decimal_overs'] > 15.0

        # Add playoff indicator (if available or use current date to estimate)
        if 'is_playoff' in live_data:
            df['is_playoff'] = live_data['is_playoff']
        else:
            # Default to False unless it's May or later (IPL playoffs typically in May)
            current_month = pd.Timestamp.now().month
            df['is_playoff'] = current_month >= 5

        # Team scores
        batting_team = df['battingTeam'].iloc[0]
        team1 = df['team1'].iloc[0]
        team2 = df['team2'].iloc[0]

        # Initialize team stats
        if batting_team == team1:
            # Team 1 is batting
            df['team1_score'] = df['currentRuns'].iloc[0]
            df['team1_wickets'] = df['currentWickets'].iloc[0]
            df['team1_runrate'] = df['currentRuns'].iloc[0] / df['decimal_overs'].iloc[0] if df['decimal_overs'].iloc[
                                                                                                 0] > 0 else 0

            if not is_second_innings:
                # First innings - team2 hasn't batted
                df['team2_score'] = 0
                df['team2_wickets'] = 0
                df['team2_runrate'] = 0
            else:
                # Second innings - team1 chasing team2's score
                df['team2_score'] = df['target'].iloc[0] - 1  # Target is team2's score + 1
                df['team2_wickets'] = 10  # Assume all out for simplicity
                df['team2_runrate'] = df['team2_score'].iloc[0] / 20  # IPL is 20 overs
        else:
            # Team 2 is batting
            df['team2_score'] = df['currentRuns'].iloc[0]
            df['team2_wickets'] = df['currentWickets'].iloc[0]
            df['team2_runrate'] = df['currentRuns'].iloc[0] / df['decimal_overs'].iloc[0] if df['decimal_overs'].iloc[
                                                                                                 0] > 0 else 0

            if not is_second_innings:
                # First innings - team1 hasn't batted
                df['team1_score'] = 0
                df['team1_wickets'] = 0
                df['team1_runrate'] = 0
            else:
                # Second innings - team2 chasing team1's score
                df['team1_score'] = df['target'].iloc[0] - 1  # Target is team1's score + 1
                df['team1_wickets'] = 10  # Assume all out for simplicity
                df['team1_runrate'] = df['team1_score'].iloc[0] / 20  # IPL is 20 overs

        # Add chase-specific features for second innings
        if is_second_innings:
            df['target'] = df['target']
            df['runs_required'] = df['target'] - df['currentRuns']
            df['balls_remaining'] = (20 - df['decimal_overs']) * 6  # IPL is 20 overs

            # Required run rate
            df['chase_required_rr'] = df['runs_required'] * 6 / df['balls_remaining'] if df['balls_remaining'].iloc[
                                                                                             0] > 0 else float('inf')
            df['current_rr'] = df['currentRuns'] / df['decimal_overs'] if df['decimal_overs'].iloc[0] > 0 else 0
            df['rr_difference'] = df['current_rr'] - df['chase_required_rr']

        # Set default values for other required features
        default_features = {
            'batting_strength': 0,
            'bowling_weakness': 0,
            'batting_team_venue_advantage': False,
            'bowling_team_venue_advantage': False
        }

        for feature, value in default_features.items():
            if feature not in df.columns:
                df[feature] = value

        # Try model prediction
        try:
            print("Making IPL win probability prediction with model...")
            win_prob = win_model.predict_proba(df)[0][1]
            print(f"Model win probability: {win_prob}")

            # Return probabilities for both teams
            batting_team = df['battingTeam'].iloc[0]
            bowling_team = df['team1'].iloc[0] if batting_team == df['team2'].iloc[0] else df['team2'].iloc[0]

            return {
                batting_team: win_prob * 100,
                bowling_team: (1 - win_prob) * 100
            }
        except Exception as e:
            print(f"Error in model prediction: {e}")
            # Fall back to IPL-specific heuristic win probability
            return calculate_ipl_win_probability(live_data)

    except Exception as e:
        print(f"Error predicting IPL winner: {e}")
        # Fall back to IPL-specific heuristic
        return calculate_ipl_win_probability(live_data)


def calculate_ipl_win_probability(live_data):
    """
    IPL-specific formula for win probability prediction when the model fails.
    Based on historical IPL patterns.

    Args:
        live_data (dict): Dictionary with current match data

    Returns:
        dict: Win probabilities for both teams
    """
    print("Using IPL-specific formula for win probability prediction...")

    # Extract needed data
    batting_team = live_data['battingTeam']
    team1 = live_data['team1']
    team2 = live_data['team2']
    bowling_team = team1 if batting_team == team2 else team2

    current_runs = live_data['currentRuns']
    current_wickets = live_data['currentWickets']

    # Parse overs
    try:
        overs_parts = str(live_data['currentOvers']).split('.')
        if len(overs_parts) == 2:
            completed_overs = int(overs_parts[0])
            balls_in_current_over = int(overs_parts[1])
        else:
            completed_overs = int(float(live_data['currentOvers']))
            balls_in_current_over = 0
    except:
        # Default if parsing fails
        completed_overs = int(float(live_data['currentOvers']))
        balls_in_current_over = 0

    # Calculate decimal overs
    decimal_overs = completed_overs + (balls_in_current_over / 6)

    # IPL is always 20 overs
    max_overs = 20
    max_balls = max_overs * 6

    # Remaining balls
    balls_consumed = decimal_overs * 6
    balls_remaining = max_balls - balls_consumed

    # Check if second innings
    is_second_innings = ('target' in live_data and live_data['target'] is not None)

    # Initialize batting win probability with a default value
    batting_win_prob = 0.5  # Default to 50% chance

    # First innings win probability calculation
    if not is_second_innings:
        # In first innings, use IPL-specific first innings formula
        innings_progress = decimal_overs / max_overs

        # Base probability - starts at 50%
        batting_win_prob = 0.5

        # IPL-specific first innings par scores at different stages
        # These are based on historical IPL data
        par_scores = {
            0.25: 45,  # 5 overs
            0.5: 85,  # 10 overs
            0.75: 135,  # 15 overs
            1.0: 170  # 20 overs
        }

        # Find the closest par score stage
        closest_stage = min(par_scores.keys(), key=lambda x: abs(x - innings_progress))
        par_score_at_stage = par_scores[closest_stage]

        # Adjust par score based on exact progress
        par_score_at_progress = par_score_at_stage
        if innings_progress < closest_stage:
            prev_stage = max([s for s in par_scores.keys() if s < closest_stage], default=0)
            prev_par = par_scores[prev_stage] if prev_stage > 0 else 0
            stage_diff = closest_stage - prev_stage
            progress_diff = innings_progress - prev_stage
            if stage_diff > 0:  # Avoid division by zero
                par_score_at_progress = prev_par + (par_score_at_stage - prev_par) * (progress_diff / stage_diff)
        elif innings_progress > closest_stage and closest_stage < 1.0:
            next_stage = min([s for s in par_scores.keys() if s > closest_stage])
            next_par = par_scores[next_stage]
            stage_diff = next_stage - closest_stage
            progress_diff = innings_progress - closest_stage
            if stage_diff > 0:  # Avoid division by zero
                par_score_at_progress = par_score_at_stage + (next_par - par_score_at_stage) * (
                            progress_diff / stage_diff)

        # Adjust based on wickets lost - IPL specific (wickets are slightly less valuable in IPL)
        wicket_factor = 1 - (current_wickets * 0.07)  # 7% reduction per wicket in IPL

        # Adjust based on run rate compared to par score
        score_factor = 1.0
        if current_runs > par_score_at_progress:
            # Above par, increase win probability (IPL teams that score above par win more often)
            score_factor = 1 + ((current_runs - par_score_at_progress) / par_score_at_progress) * 0.6
        else:
            # Below par, decrease win probability
            score_factor = 1 - ((par_score_at_progress - current_runs) / par_score_at_progress) * 0.5

        # IPL-specific: toss advantage is significant in IPL
        toss_factor = 1.05 if live_data.get('tossWinner') == batting_team else 0.95

        # IPL-specific: venue advantage
        venue_factor = 1.0
        if 'venue' in live_data:
            venue = live_data['venue'].lower()
            # Team home ground advantage
            team_venues = {
                'mumbai indians': ['wankhede', 'brabourne'],
                'chennai super kings': ['chepauk'],
                'royal challengers bangalore': ['chinnaswamy'],
                'kolkata knight riders': ['eden gardens'],
                'delhi capitals': ['feroz shah kotla', 'arun jaitley'],
                'punjab kings': ['mohali', 'punjab cricket association'],
                'rajasthan royals': ['sawai mansingh'],
                'sunrisers hyderabad': ['rajiv gandhi', 'uppal']
            }

            for team, venues in team_venues.items():
                if any(v in venue.lower() for v in venues):
                    if team.lower() in batting_team.lower():
                        venue_factor = 1.08  # 8% home advantage for batting team
                    elif team.lower() in bowling_team.lower():
                        venue_factor = 0.92  # 8% home advantage for bowling team

        # Combine all IPL-specific factors
        batting_win_prob = batting_win_prob * wicket_factor * score_factor * toss_factor * venue_factor

        # Ensure probability is between 15% and 85% for first innings
        batting_win_prob = max(0.15, min(0.85, batting_win_prob))

    else:
        # Second innings - IPL-specific chase parameters
        target = live_data['target']
        runs_required = target - current_runs

        if runs_required <= 0:
            # Already won
            return {batting_team: 99, bowling_team: 1}

        # Required run rate
        required_rr = (runs_required * 6) / balls_remaining if balls_remaining > 0 else float('inf')

        # Current run rate
        current_rr = current_runs / balls_consumed if balls_consumed > 0 else 0

        # IPL-specific chase statistics based on historical data

        # Factor 1: IPL teams are better at chasing - Run rate comparison
        rr_factor = current_rr / required_rr if required_rr > 0 else 2.0
        rr_factor = min(2.0, rr_factor)  # Cap at 2.0

        # Factor 2: Wickets are less crucial in IPL chases due to batting depth
        wickets_in_hand = 10 - current_wickets
        wicket_factor = (wickets_in_hand / 4)  # Scale wickets (less impactful in IPL)

        # Factor 3: Balls remaining and match progression
        progress_factor = 1.0 - (balls_remaining / max_balls)  # How far into the chase

        # IPL-specific: Many matches decided in last over
        last_over_factor = 1.0
        if completed_overs >= 19:  # Last over
            # If it's close in the last over, give slight edge to batting team
            if runs_required <= 12:  # 2 sixes or less needed
                last_over_factor = 1.2

        # IPL-specific: Death over specialists make a difference
        death_bowling_factor = 1.0
        if completed_overs >= 16:  # Death overs
            # Adjust based on death bowling strength if known
            # This would ideally come from team data
            death_bowling_factor = 0.95  # Slight advantage to bowling team in death

        # Combine factors - weighted sum with IPL-specific weights
        chase_difficulty = (0.45 * rr_factor) + \
                           (0.25 * wicket_factor) + \
                           (0.15 * (1 - progress_factor)) + \
                           (0.15 * last_over_factor * death_bowling_factor)  # Specialist impact

        # Convert to probability
        # chase_difficulty: 0 = impossible, 1 = even chance, 2 = very easy
        batting_win_prob = chase_difficulty / 2

        # IPL-specific adjustments for extreme scenarios
        if required_rr > 15 and balls_remaining > 12:
            batting_win_prob *= 0.7  # Hard but not impossible in IPL
        elif required_rr > 20:
            batting_win_prob *= 0.3  # Very difficult even in IPL

        # Toss factor in IPL chases
        if live_data.get('tossWinner') == batting_team:
            batting_win_prob *= 1.07  # 7% boost for toss winners in IPL chases

        # Ensure probability is between 1% and 99%
        batting_win_prob = max(0.01, min(0.99, batting_win_prob))

    # Return probabilities for both teams
    return {
        batting_team: batting_win_prob * 100,
        bowling_team: (1 - batting_win_prob) * 100
    }


# Function for live IPL match prediction
def live_prediction(live_data, score_model=None, win_model=None):
    """
    Makes predictions based on live IPL match data.

    Args:
        live_data (dict): Dictionary containing live IPL match data with the following keys:
                         team1, team2, battingTeam, tossWinner, tossDecision,
                         currentRuns, currentWickets, currentOvers, venue (optional),
                         target (optional), is_playoff (optional)
        score_model: The trained model for IPL score prediction
        win_model: The trained model for IPL win prediction

    Returns:
        dict: Dictionary containing estimated score and win probabilities
    """
    # Load models if not provided
    if score_model is None or win_model is None:
        score_model, win_model = load_models()

    # Predict estimated final score
    estimated_score = predict_final_score(live_data, score_model)

    # Predict match winner probability
    win_probabilities = predict_winner_probability(live_data, win_model)

    # Return predictions
    return {
        'estimated_score': estimated_score,
        'win_probabilities': win_probabilities
    }


# Example usage
if __name__ == "__main__":
    # Load models
    score_model, win_model = load_models()

    # Get live match data from user input
    print("Enter live IPL match data:")
    team1 = input("Team 1: ")
    team2 = input("Team 2: ")
    batting_team = input("Current Batting Team: ")
    toss_winner = input("Toss Winner: ")
    toss_decision = input("Toss Decision (bat/field): ")
    current_runs = int(input("Current Runs: "))
    current_wickets = int(input("Current Wickets: "))
    current_overs = input("Current Overs (format: X.Y): ")
    venue = input("Venue (optional): ")

    # Ask if this is a playoff match
    is_playoff = input("Is this a playoff match? (y/n): ").lower().startswith('y')

    # Ask for target if it's the second innings
    is_second_innings = input("Is this the second innings? (y/n): ").lower().startswith('y')
    target = int(input("Target: ")) if is_second_innings else None

    # Create live data dictionary
    live_data = {
        'team1': team1,
        'team2': team2,
        'battingTeam': batting_team,
        'tossWinner': toss_winner,
        'tossDecision': toss_decision,
        'currentRuns': current_runs,
        'currentWickets': current_wickets,
        'currentOvers': current_overs,
        'is_playoff': is_playoff
    }

    # Add optional fields
    if venue:
        live_data['venue'] = venue

    if target:
        live_data['target'] = target

    # Make predictions
    predictions = live_prediction(live_data, score_model, win_model)

    # Print predictions
    print("\nIPL Match Predictions:")
    print(f"Estimated Final Score: {predictions['estimated_score']}")
    print("\nWin Probabilities:")
    for team, probability in predictions['win_probabilities'].items():
        print(f"{team}: {probability:.1f}%")