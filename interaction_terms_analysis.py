import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def main():
    print("Analyzing Interaction Terms in Housing Price Prediction")
    
    # Load training data
    train_path = "train_housingprices.csv"
    feature_cols = ['GrLivArea', 'LotArea', 'YearBuilt', 'OverallQual']
    target_col = 'SalePrice'
    
    print("\nLoading training data...")
    train_data = pd.read_csv(train_path)
    
    # Extract features and target
    X = train_data[feature_cols].values
    y = train_data[target_col].values
    
    print(f"Loaded {len(X)} training samples")
    
    # Normalize data to prevent numerical issues
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std = np.maximum(X_std, 1e-8)  # Prevent division by zero
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std
    
    # Generate polynomial features with interaction terms
    print("\nGenerating polynomial features with interaction terms...")
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
    X_poly = poly.fit_transform(X_norm)
    
    # Get feature names
    feature_names = poly.get_feature_names_out(feature_cols)
    print(f"Generated {len(feature_names)} polynomial features")
    
    # Fit linear regression model
    print("\nFitting linear regression model...")
    model = LinearRegression()
    model.fit(X_poly, y_norm)
    
    # Get coefficients
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Create DataFrame with feature names and coefficients
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    
    # Add intercept row
    intercept_df = pd.DataFrame({
        'Feature': ['intercept'],
        'Coefficient': [intercept]
    })
    
    coef_df = pd.concat([intercept_df, coef_df], ignore_index=True)
    
    # Sort by absolute coefficient value
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
    
    # Print all interaction terms
    print("\nCoefficients for interaction terms:")
    interaction_terms = [term for term in feature_names if ' ' in term]
    for term in interaction_terms:
        idx = list(feature_names).index(term)
        coef = coefficients[idx]
        print(f"{term}: {coef:.6f}")
    
    # Print top 20 most important features
    print("\nTop 20 most important features:")
    print(coef_df.head(20))
    
    # Save all coefficients to CSV
    coef_df.to_csv("polynomial_interaction_coefficients.csv", index=False)
    print("\nSaved all coefficients to polynomial_interaction_coefficients.csv")
    
    # Plot feature importance for top 15 features
    plt.figure(figsize=(12, 8))
    top_features = coef_df.head(15)
    plt.barh(top_features['Feature'], top_features['Abs_Coefficient'])
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Top 15 Feature Importance')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top
    plt.tight_layout()
    plt.savefig('polynomial_interaction_importance.png')
    print("\nSaved feature importance plot to polynomial_interaction_importance.png")
    
    # Specifically highlight GrLivArea*OverallQual interaction
    target_interaction = "GrLivArea OverallQual"
    if target_interaction in feature_names:
        idx = list(feature_names).index(target_interaction)
        coef = coefficients[idx]
        print(f"\nCoefficient for {target_interaction}: {coef:.6f}")
        
        # Calculate relative importance
        abs_coefs = np.abs(coefficients)
        importance = abs_coefs[idx] / np.sum(abs_coefs) * 100
        print(f"Relative importance: {importance:.2f}%")
        
        # Rank among all features
        rank = coef_df[coef_df['Feature'] == target_interaction].index[0] + 1
        print(f"Rank among all features: {rank}")
    else:
        print(f"\nInteraction term {target_interaction} not found in feature names")

if __name__ == "__main__":
    main()
