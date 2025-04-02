import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ipl_live_prediction import live_prediction

# Set page configuration
st.set_page_config(page_title="IPL Match Prediction", layout="wide")
st.title("IPL 2025 Match Prediction")
st.markdown("Predict estimated innings score and match winner probability based on live IPL match data.")

# Add IPL logo/branding
try:
    st.sidebar.image("https://static.iplt20.com/players/284/ipl-logo-new-old.png", width=150)
except:
    pass
st.sidebar.markdown("### IPL 2025 Season")

# Check if models exist
score_model_path = 'ipl_score_prediction_model.joblib'
win_model_path = 'ipl_win_prediction_model.joblib'

if not (os.path.exists(score_model_path) and os.path.exists(win_model_path)):
    st.warning("IPL prediction models not found. Using fallback prediction methods.")
    score_model = None
    win_model = None
else:
    # Load models
    try:
        score_model = joblib.load(score_model_path)
        win_model = joblib.load(win_model_path)
        st.success("IPL prediction models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        score_model = None
        win_model = None
        st.info("Continuing with IPL-specific fallback prediction methods.")

# Create input form
st.header("Match Information")
col1, col2 = st.columns(2)

with col1:
    # IPL teams dropdown
    ipl_teams = [
        "",
        "Mumbai Indians",
        "Chennai Super Kings",
        "Royal Challengers Bangalore",
        "Kolkata Knight Riders",
        "Delhi Capitals",
        "Sunrisers Hyderabad",
        "Punjab Kings",
        "Rajasthan Royals",
        "Gujarat Titans",
        "Lucknow Super Giants"
    ]

    team1 = st.selectbox("Team 1", options=ipl_teams)
    team2 = st.selectbox("Team 2", options=[t for t in ipl_teams if t != team1 or t == ""])

    batting_team = st.selectbox("Current Batting Team",
                                options=["", team1, team2],
                                disabled=not (team1 and team2))

    toss_winner = st.selectbox("Toss Winner",
                               options=["", team1, team2],
                               disabled=not (team1 and team2))

    toss_decision = st.selectbox("Toss Decision", options=["", "bat", "field"])

    # Add IPL venue selection
    ipl_venues = [
        "",
        "M.A. Chidambaram Stadium, Chennai",
        "Wankhede Stadium, Mumbai",
        "Eden Gardens, Kolkata",
        "M. Chinnaswamy Stadium, Bengaluru",
        "Arun Jaitley Stadium, Delhi",
        "Rajiv Gandhi International Stadium, Hyderabad",
        "Punjab Cricket Association Stadium, Mohali",
        "Narendra Modi Stadium, Ahmedabad",
        "Sawai Mansingh Stadium, Jaipur",
        "Brabourne Stadium, Mumbai",
        "Dr DY Patil Sports Academy, Mumbai",
        "Maharashtra Cricket Association Stadium, Pune",
        "BRSABV Ekana Cricket Stadium, Lucknow",
        "Other"
    ]
    venue_selection = st.selectbox("Venue", options=ipl_venues)

    # Allow custom venue if "Other" is selected
    venue = venue_selection
    if venue_selection == "Other":
        venue = st.text_input("Enter Venue Name")

    # Is this a playoff match?
    is_playoff = st.checkbox("Playoff Match")

with col2:
    current_runs = st.number_input("Current Runs", min_value=0, step=1)
    current_wickets = st.number_input("Current Wickets", min_value=0, max_value=10, step=1)
    current_overs = st.text_input("Current Overs (format: X.Y)")

    is_second_innings = st.checkbox("Second Innings")
    target = None
    if is_second_innings:
        target = st.number_input("Target", min_value=1, step=1)

# Make prediction button
if st.button("Make Prediction", type="primary"):
    # Validate inputs
    if not (team1 and team2 and batting_team and toss_winner and toss_decision and current_overs):
        st.error("Please fill all the required fields.")
    else:
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
        if venue and venue != "":
            live_data['venue'] = venue

        if is_second_innings and target:
            live_data['target'] = target

        # Make predictions
        with st.spinner("Making IPL match predictions..."):
            predictions = live_prediction(live_data, score_model, win_model)

        # Create tabs for different prediction displays
        tab1, tab2 = st.tabs(["Basic Predictions", "Detailed Analysis"])

        with tab1:
            # Display basic predictions
            st.header("Predictions")

            # Show estimated score
            est_score = predictions['estimated_score']
            if is_second_innings and target:
                st.metric("Estimated Final Score", est_score,
                          delta=f"{est_score - target} from target" if est_score != target else "Exact target")
            else:
                st.metric("Estimated Final Score", est_score)

            # Show win probabilities
            st.subheader("Win Probabilities")

            # Create probability dataframe for display
            win_probs = predictions['win_probabilities']
            prob_df = pd.DataFrame({
                'Team': list(win_probs.keys()),
                'Win Probability (%)': list(win_probs.values())
            })

            # Display as a table
            st.table(prob_df)

            # Display as a bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = sns.barplot(x='Team', y='Win Probability (%)', data=prob_df, ax=ax)

            # Add text labels on bars
            for i, p in enumerate(bars.patches):
                bars.annotate(f"{p.get_height():.1f}%",
                              (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center', va='bottom')

            plt.title('Win Probability')
            plt.ylim(0, 100)
            st.pyplot(fig)

        with tab2:
            # Display detailed analysis
            st.header("Detailed Match Analysis")

            # If it's the second innings, show chase analysis
            if is_second_innings and target:
                st.subheader("Chase Analysis")

                # Calculate required run rate
                try:
                    overs_parts = current_overs.split('.')
                    completed_overs = int(overs_parts[0])
                    balls_in_current_over = int(overs_parts[1]) if len(overs_parts) > 1 else 0

                    decimal_overs = completed_overs + balls_in_current_over / 6
                    max_overs = 20  # IPL is T20

                    remaining_overs = max_overs - decimal_overs
                    remaining_balls = remaining_overs * 6
                    runs_required = target - current_runs

                    if remaining_balls > 0:
                        required_rr = (runs_required * 6) / remaining_balls
                        current_rr = current_runs / decimal_overs if decimal_overs > 0 else 0

                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Runs Required", runs_required)
                        with col2:
                            st.metric("Balls Remaining", int(remaining_balls))
                        with col3:
                            st.metric("Required Run Rate", f"{required_rr:.2f}",
                                      delta=f"{required_rr - current_rr:.2f} from current RR")

                        # Add winning scenarios
                        st.subheader("Winning Scenarios")

                        # Calculate win percentage based on different criteria
                        wickets_lost = current_wickets
                        wickets_left = 10 - wickets_lost

                        # Create a table of scenarios
                        scenarios = []

                        # Scenario 1: Current run rate maintained
                        current_rate_score = current_runs + (current_rr * remaining_overs)
                        current_rate_result = "Win" if current_rate_score >= target else "Lose"
                        current_rate_margin = current_rate_score - target if current_rate_result == "Win" else target - current_rate_score
                        scenarios.append({
                            "Scenario": "Maintain current run rate",
                            "Run Rate": f"{current_rr:.2f}",
                            "Projected Score": f"{int(current_rate_score)}",
                            "Result": current_rate_result,
                            "Margin": f"{int(current_rate_margin)}"
                        })

                        # Scenario 2: Required run rate
                        scenarios.append({
                            "Scenario": "Required run rate",
                            "Run Rate": f"{required_rr:.2f}",
                            "Projected Score": f"{target}",
                            "Result": "Win",
                            "Margin": "0"
                        })

                        # Scenario 3: Aggressive (120% of required)
                        aggressive_rr = required_rr * 1.2
                        aggressive_score = current_runs + (aggressive_rr * remaining_overs / 6)
                        aggressive_margin = int(aggressive_score - target)
                        scenarios.append({
                            "Scenario": "Aggressive (120% of required)",
                            "Run Rate": f"{aggressive_rr:.2f}",
                            "Projected Score": f"{int(aggressive_score)}",
                            "Result": "Win",
                            "Margin": f"{aggressive_margin}"
                        })

                        # Display scenarios
                        st.table(pd.DataFrame(scenarios))

                        # Show a run rate comparison chart
                        run_rates = {
                            "Current RR": current_rr,
                            "Required RR": required_rr
                        }

                        rr_df = pd.DataFrame({
                            'Type': list(run_rates.keys()),
                            'Run Rate': list(run_rates.values())
                        })

                        fig, ax = plt.subplots(figsize=(8, 4))
                        bars = sns.barplot(x='Type', y='Run Rate', data=rr_df, ax=ax)

                        # Add text labels on bars
                        for i, p in enumerate(bars.patches):
                            bars.annotate(f"{p.get_height():.2f}",
                                          (p.get_x() + p.get_width() / 2., p.get_height()),
                                          ha='center', va='bottom')

                        plt.title('Run Rate Comparison')
                        st.pyplot(fig)

                        # Add IPL-specific insights
                        st.subheader("IPL-Specific Insights")

                        # Compare to historical successful chases
                        st.write("#### Historical Chase Success in IPL")

                        # These are approximate stats based on IPL history
                        if required_rr <= 8:
                            st.success(
                                f"🏆 Historically, teams successfully chase targets requiring RRR ≤ 8 about 75% of the time in IPL.")
                        elif required_rr <= 10:
                            st.info(
                                f"🏏 Chases requiring RRR between 8-10 have been successful about 50% of the time in IPL.")
                        elif required_rr <= 12:
                            st.warning(
                                f"⚠️ Chases requiring RRR between 10-12 have been successful about 30% of the time in IPL.")
                        elif required_rr <= 15:
                            st.warning(
                                f"⚠️ Chases requiring RRR between 12-15 have been successful about 15% of the time in IPL.")
                        else:
                            st.error(
                                f"🛑 Chases requiring RRR > 15 have been successful less than 5% of the time in IPL history.")

                        # Team-specific insights
                        st.write("#### Team Insights")

                        # Add team-specific insights if available
                        batting_team_lower = batting_team.lower()

                        # These would ideally come from a database of team stats
                        if "mumbai" in batting_team_lower:
                            st.info(
                                "📊 Mumbai Indians have historically been one of the best chasing teams in IPL, especially in death overs.")
                        elif "chennai" in batting_team_lower:
                            st.info("📊 Chennai Super Kings have a strong record of successful chases at home venues.")
                        elif "bangalore" in batting_team_lower:
                            st.info(
                                "📊 Royal Challengers Bangalore have the batting depth to achieve high run rates in chases.")
                        elif "kolkata" in batting_team_lower:
                            st.info("📊 Kolkata Knight Riders have a good record of successful chases at Eden Gardens.")

                except Exception as e:
                    st.error(f"Error in chase analysis: {str(e)}")
            else:
                # For first innings, show projected score analysis
                st.subheader("Projected Score Analysis")

                try:
                    overs_parts = current_overs.split('.')
                    completed_overs = int(overs_parts[0])
                    balls_in_current_over = int(overs_parts[1]) if len(overs_parts) > 1 else 0

                    decimal_overs = completed_overs + balls_in_current_over / 6

                    remaining_overs = 20 - decimal_overs  # IPL is T20
                    current_rr = current_runs / decimal_overs if decimal_overs > 0 else 0

                    # Create a table of projection scenarios
                    projections = []

                    # Scenario 1: Current run rate maintained
                    current_rate_score = current_runs + (current_rr * remaining_overs)
                    projections.append({
                        "Scenario": "Maintain current run rate",
                        "Run Rate": f"{current_rr:.2f}",
                        "Projected Score": f"{int(current_rate_score)}"
                    })

                    # Scenario 2: Improved run rate (120%)
                    improved_rr = current_rr * 1.2
                    improved_score = current_runs + (improved_rr * remaining_overs)
                    projections.append({
                        "Scenario": "Accelerate (120% of current)",
                        "Run Rate": f"{improved_rr:.2f}",
                        "Projected Score": f"{int(improved_score)}"
                    })

                    # Scenario 3: Death overs acceleration (150%)
                    death_rr = current_rr * 1.5
                    death_score = current_runs + (death_rr * remaining_overs)
                    projections.append({
                        "Scenario": "Death overs boost (150% of current)",
                        "Run Rate": f"{death_rr:.2f}",
                        "Projected Score": f"{int(death_score)}"
                    })

                    # Scenario 4: Fall of wickets (80%)
                    wicket_rr = current_rr * 0.8
                    wicket_score = current_runs + (wicket_rr * remaining_overs)
                    projections.append({
                        "Scenario": "Lose wickets (80% of current)",
                        "Run Rate": f"{wicket_rr:.2f}",
                        "Projected Score": f"{int(wicket_score)}"
                    })

                    # Display projections
                    st.table(pd.DataFrame(projections))

                    # Show a projected scores chart
                    scores = {
                        "Current Run Rate": int(current_rate_score),
                        "Model Prediction": est_score,
                        "Accelerated": int(improved_score)
                    }

                    score_df = pd.DataFrame({
                        'Scenario': list(scores.keys()),
                        'Projected Score': list(scores.values())
                    })

                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = sns.barplot(x='Scenario', y='Projected Score', data=score_df, ax=ax)

                    # Add text labels on bars
                    for i, p in enumerate(bars.patches):
                        bars.annotate(f"{p.get_height():.0f}",
                                      (p.get_x() + p.get_width() / 2., p.get_height()),
                                      ha='center', va='bottom')

                    plt.title('Projected Final Scores')
                    st.pyplot(fig)

                    # Add IPL-specific insights
                    st.subheader("IPL-Specific Insights")

                    # Phase-wise analysis
                    st.write("#### Phase Analysis")

                    # Determine current phase
                    phase = "Powerplay (1-6)" if decimal_overs <= 6 else "Middle Overs (7-15)" if decimal_overs <= 15 else "Death Overs (16-20)"

                    # Add phase-specific insights
                    if phase == "Powerplay (1-6)":
                        st.info(f"📊 Current phase: {phase}. Teams in IPL typically score at 8-9 RPO during powerplay.")
                        avg_powerplay = 50  # Approximate
                        if current_runs > avg_powerplay * (decimal_overs / 6):
                            st.success(f"✅ Above average powerplay score for this stage.")
                        else:
                            st.warning(f"⚠️ Below average powerplay score for this stage.")
                    elif phase == "Middle Overs (7-15)":
                        st.info(
                            f"📊 Current phase: {phase}. Teams in IPL typically score at 7-8 RPO during middle overs.")
                        expected_score = 50 + (decimal_overs - 6) * 7.5  # Rough approximation
                        if current_runs > expected_score:
                            st.success(f"✅ Above average score for this stage of the innings.")
                        else:
                            st.warning(f"⚠️ Below average score for this stage of the innings.")
                    else:  # Death overs
                        st.info(
                            f"📊 Current phase: {phase}. Teams in IPL typically score at 10-12 RPO during death overs.")
                        expected_score = 120 + (decimal_overs - 15) * 11  # Rough approximation
                        if current_runs > expected_score:
                            st.success(f"✅ Above average score for this stage of the innings.")
                        else:
                            st.warning(f"⚠️ Below average score for this stage of the innings.")

                    # Venue-specific insights
                    if venue and venue != "":
                        st.write("#### Venue Analysis")
                        venue_lower = venue.lower()

                        # These would ideally come from a database of venue stats
                        if any(v in venue_lower for v in ['chinnaswamy', 'bengaluru', 'bangalore']):
                            st.info(
                                "🏟️ M. Chinnaswamy Stadium is known for high scores. Average first innings score: 180-190")
                        elif any(v in venue_lower for v in ['wankhede', 'mumbai']):
                            st.info(
                                "🏟️ Wankhede Stadium typically has good batting conditions. Average first innings score: 175-185")
                        elif any(v in venue_lower for v in ['chepauk', 'chennai']):
                            st.info(
                                "🏟️ M.A. Chidambaram Stadium tends to favor spinners. Average first innings score: 165-175")
                        elif any(v in venue_lower for v in ['eden', 'kolkata']):
                            st.info("🏟️ Eden Gardens has balanced conditions. Average first innings score: 170-180")
                        elif any(v in venue_lower for v in ['kotla', 'delhi', 'arun jaitley']):
                            st.info(
                                "🏟️ Arun Jaitley Stadium typically has slow pitches. Average first innings score: 165-175")

                    # Team-specific insights
                    st.write("#### Team Batting Analysis")

                    # Add team-specific insights if available
                    batting_team_lower = batting_team.lower()

                    # These would ideally come from a database of team stats
                    if "mumbai" in batting_team_lower:
                        st.info("📊 Mumbai Indians typically accelerate well in death overs with strong finishers.")
                    elif "chennai" in batting_team_lower:
                        st.info("📊 Chennai Super Kings often maintain a steady run rate throughout their innings.")
                    elif "bangalore" in batting_team_lower:
                        st.info(
                            "📊 Royal Challengers Bangalore tend to have explosive batting performances, especially at home.")
                    elif "kolkata" in batting_team_lower:
                        st.info("📊 Kolkata Knight Riders are known for aggressive batting in powerplay overs.")

                except Exception as e:
                    st.error(f"Error in projection analysis: {str(e)}")

if __name__ == "__main__":
    # This will be executed when running the Streamlit app
    pass