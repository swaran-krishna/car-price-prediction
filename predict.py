import joblib
import pandas as pd
import numpy as np

# Load the saved model (includes the fitted preprocessor)
model = joblib.load('../../backend/car_price_model.pkl')


def predict_price(input_data):
    """
    Predict the car price based on input data.

    Args:
        input_data (dict): Dictionary containing the input features.

    Returns:
        float: Predicted selling price.
    """
    try:
        # Convert input dictionary to DataFrame
        new_data = pd.DataFrame([input_data])

        # Calculate log-transformed features for 'mmr' and 'odometer'
        new_data['log_mmr'] = np.log1p(new_data['mmr'])
        new_data['log_odometer'] = np.log1p(new_data['odometer'])

        # Ensure the DataFrame has all expected columns in the correct order
        expected_columns = ['year', 'odometer', 'mmr', 'car_age', 'miles_per_year',
                            'make', 'model', 'condition', 'transmission', 'state',
                            'color', 'seller', 'log_mmr', 'log_odometer']
        new_data = new_data.reindex(columns=expected_columns, fill_value='Unknown')

        # Predict (the pipeline handles preprocessing)
        prediction = model.predict(new_data)[0]
        prediction = max(prediction, 1000)  # Ensure prediction is at least 1000

        return float(prediction)

    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")


# Example usage (for testing purposes)
if __name__ == "__main__":
    # Sample input data
    sample_data = {
        'year': 2015,
        'odometer': 50000,
        'mmr': 15000,
        'car_age': 10,
        'miles_per_year': 5000,
        'make': 'Toyota',
        'model': 'Camry',
        'condition': 'Good',
        'transmission': 'automatic',
        'state': 'CA',
        'color': 'black',
        'seller': 'dealer'
    }

    # Predict and print the result
    predicted_price = predict_price(sample_data)
    print("Predicted Selling Price:", predicted_price)