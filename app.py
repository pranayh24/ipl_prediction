from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib
from ipl_live_prediction import live_prediction
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load models on startup
score_model = None
win_model = None


def load_models():
    global score_model, win_model
    try:
        logger.info("Loading IPL prediction models...")
        score_model = joblib.load('ipl_score_prediction_model.joblib')
        win_model = joblib.load('ipl_win_prediction_model.joblib')
        logger.info("IPL models loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Error loading IPL models: {e}")
        return False


# API endpoint for IPL predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    global score_model, win_model

    # Check if models are loaded
    if score_model is None or win_model is None:
        if not load_models():
            return jsonify({
                "error": "Models could not be loaded. Using fallback methods."
            })

    try:
        # Get data from request
        live_data = request.json
        logger.info(f"Received prediction request with data: {live_data}")

        # Validate required fields
        required_fields = ['team1', 'team2', 'battingTeam', 'tossWinner',
                           'tossDecision', 'currentRuns', 'currentWickets', 'currentOvers']

        for field in required_fields:
            if field not in live_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Make predictions
        predictions = live_prediction(live_data, score_model, win_model)

        # Return predictions
        response = {
            "estimated_score": predictions['estimated_score'],
            "win_probabilities": {
                k: round(v, 2) for k, v in predictions['win_probabilities'].items()
            }
        }

        # For second innings, add chase-specific information
        if 'target' in live_data and live_data['target'] is not None:
            target = live_data['target']
            current_runs = live_data['currentRuns']

            # Calculate overs
            try:
                overs_parts = live_data['currentOvers'].split('.')
                if len(overs_parts) == 2:
                    completed_overs = int(overs_parts[0])
                    balls_in_current_over = int(overs_parts[1])
                else:
                    completed_overs = int(float(live_data['currentOvers']))
                    balls_in_current_over = 0
            except:
                completed_overs = int(float(live_data['currentOvers']))
                balls_in_current_over = 0

            decimal_overs = completed_overs + (balls_in_current_over / 6)
            remaining_overs = 20 - decimal_overs  # IPL is T20

            runs_required = target - current_runs
            required_rr = runs_required / remaining_overs if remaining_overs > 0 else float('inf')

            # Add chase info to response
            response["chase_info"] = {
                "target": target,
                "runs_required": runs_required,
                "balls_remaining": int(remaining_overs * 6),
                "required_run_rate": round(required_rr, 2)
            }

        logger.info(f"Prediction response: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({"error": str(e)}), 500


# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})


if __name__ == '__main__':
    # Load models on startup
    load_models()

    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))

    # Run the Flask app
    app.run(host='0.0.0.0', port=port)