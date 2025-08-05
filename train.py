import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

np.random.seed(42)
sns.set_style('whitegrid')

# -------- Load and Process Dataset --------
def load_and_process_dataset(filepath):
    df = pd.read_csv(filepath)
    df = df.rename(columns={'colour': 'color', 'selling price': 'sellingprice'})

    required_columns = ['odometer', 'year', 'sellingprice', 'mmr', 'make', 'model', 'condition',
                        'transmission', 'state', 'color', 'seller']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Fill missing values
    df['odometer'] = df['odometer'].fillna(df['odometer'].median())
    df['year'] = df['year'].fillna(df['year'].median())
    df['mmr'] = df['mmr'].fillna(df['mmr'].median())
    for col in ['make', 'model', 'condition', 'transmission', 'state', 'color', 'seller']:
        df[col] = df[col].fillna('Unknown')

    current_year = datetime.now().year
    df = df[df['year'].between(1900, current_year)]

    valid_conditions = ['Excellent', 'Good', 'Fair', 'Poor']
    df['condition'] = df['condition'].apply(lambda x: x if x in valid_conditions else 'Good')

    # Feature Engineering
    df['car_age'] = current_year - df['year']
    df['miles_per_year'] = df['odometer'] / (df['car_age'] + 1)

    # Log-transform skewed features
    df['log_mmr'] = np.log1p(df['mmr'])
    df['log_odometer'] = np.log1p(df['odometer'])

    # Impute selling price if needed
    if df['sellingprice'].isna().any():
        condition_multiplier = {'Excellent': 1.1, 'Good': 1.0, 'Fair': 0.9, 'Poor': 0.7}
        base_price = df['mmr'] * np.random.uniform(0.8, 1.2, len(df))
        df['sellingprice'] = [base_price[i] * condition_multiplier[df['condition'][i]] for i in range(len(df))]

    # Reduce cardinality
    for col in ['model', 'seller']:
        top_100 = df[col].value_counts().nlargest(100).index
        df[col] = df[col].where(df[col].isin(top_100), 'Other')

    return df

# -------- Visualizations --------
def plot_data_visualizations(df, numerical_cols, y_test, y_pred, best_model):
    viz_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../visualizations'))
    os.makedirs(viz_dir, exist_ok=True)

    # Histograms
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'numerical_features_histogram.png'))
    plt.close()

    # Boxplot by condition
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='condition', y='sellingprice', data=df, order=['Excellent', 'Good', 'Fair', 'Poor'])
    plt.title('Selling Price by Condition')
    plt.savefig(os.path.join(viz_dir, 'selling_price_by_condition.png'))
    plt.close()

    # Correlation heatmap
    corr_cols = numerical_cols + ['sellingprice']
    correlation_matrix = df[corr_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(viz_dir, 'correlation_heatmap.png'))
    plt.close()

    # Scatter plot matrix
    fig = px.scatter_matrix(df, dimensions=numerical_cols, title='Scatter Matrix', height=800)
    fig.update_traces(diagonal_visible=False)
    fig.write_html(os.path.join(viz_dir, 'scatter_matrix.html'))

    # Boxplot by make
    top_makes = df['make'].value_counts().head(10).index
    df_filtered = df[df['make'].isin(top_makes)]
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='make', y='sellingprice', data=df_filtered)
    plt.title('Selling Price by Make (Top 10)')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(viz_dir, 'price_by_make_boxplot.png'), bbox_inches='tight')
    plt.close()

    # Violin plot by transmission
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='transmission', y='sellingprice', data=df)
    plt.title('Selling Price by Transmission')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(viz_dir, 'price_by_transmission_violin.png'), bbox_inches='tight')
    plt.close()

    # Actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Actual vs Predicted')
    plt.savefig(os.path.join(viz_dir, 'actual_vs_predicted.png'))
    plt.close()

    # Feature importances
    feature_importance = best_model.named_steps['regressor'].feature_importances_
    feature_names = (
        numerical_cols +
        best_model.named_steps['preprocessor']
            .named_transformers_['cat']
            .named_steps['encoder']
            .get_feature_names_out(categorical_cols).tolist()
    )
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    top_10 = importance_df.sort_values('Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_10)
    plt.title('Top 10 Feature Importances')
    plt.savefig(os.path.join(viz_dir, 'feature_importance.png'))
    plt.close()

# -------- Main Execution --------
# Load data
df = load_and_process_dataset(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'car_prices.csv'))
)

# Define features
numerical_cols = ['log_odometer', 'log_mmr', 'car_age', 'miles_per_year']
categorical_cols = ['make', 'model', 'condition', 'transmission', 'state', 'color', 'seller']
X = df[numerical_cols + categorical_cols]
y = df['sellingprice']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Model pipeline and tuning
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
])
param_grid = {
    'regressor__learning_rate': [0.01, 0.1],
    'regressor__max_depth': [3, 5],
    'regressor__n_estimators': [50, 100]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)
print("✅ Best Parameters:", grid_search.best_params_)
print(f"✅ MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"✅ RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"✅ R²: {r2_score(y_test, y_pred):.2f}")

# Visualize
plot_data_visualizations(df, numerical_cols, y_test, y_pred, best_model)

# Save artifacts
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend'))
os.makedirs(backend_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(backend_dir, 'car_price_model.pkl'))
joblib.dump({
    'best_params': grid_search.best_params_,
    'features': numerical_cols + categorical_cols,
    'timestamp': datetime.now().isoformat(),
    'metrics': {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
}, os.path.join(backend_dir, 'car_price_metadata.pkl'))
joblib.dump(best_model.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler'], os.path.join(backend_dir, 'scaler.pkl'))
joblib.dump(best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'], os.path.join(backend_dir, 'encoder.pkl'))
print("✅ Artifacts saved in backend folder")
