import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from housing_ridge_regression import RidgeRegression, load_housing_data

def analyze_feature_importance(model, feature_names, top_n=20):
    """
    Analyze and display feature importance from a trained ridge regression model
    
    Parameters:
    -----------
    model : RidgeRegression
        Trained ridge regression model
    feature_names : list
        Names of features
    top_n : int, default=20
        Number of top features to display
    """
    # Create a DataFrame for easier sorting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': model.weights,
        'Absolute_Importance': np.abs(model.weights)
    })
    
    # Sort by absolute importance
    importance_df = importance_df.sort_values('Absolute_Importance', ascending=False)
    
    # Print top features
    print("\n===== FEATURE IMPORTANCE ANALYSIS =====")
    print(f"Top {top_n} most important features:")
    print("-" * 50)
    for i, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
        print(f"{i}. {row['Feature']}: {row['Weight']:.6f} (abs: {row['Absolute_Importance']:.6f})")
    
    # Plot feature importance
    plt.figure(figsize=(12, 10))
    plt.barh(importance_df.head(top_n)['Feature'], importance_df.head(top_n)['Absolute_Importance'])
    plt.xlabel('Absolute Weight')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importance in Ridge Regression')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top
    plt.tight_layout()
    plt.savefig('ridge_feature_importance.png')
    print(f"Feature importance plot saved to ridge_feature_importance.png")
    
    # Analyze numerical features separately
    numerical_features = [f for f in feature_names if not '_' in f]
    num_importance_df = importance_df[importance_df['Feature'].isin(numerical_features)]
    
    print("\n===== NUMERICAL FEATURE IMPORTANCE =====")
    print(f"Top {min(top_n, len(numerical_features))} most important numerical features:")
    print("-" * 50)
    for i, (_, row) in enumerate(num_importance_df.head(top_n).iterrows(), 1):
        print(f"{i}. {row['Feature']}: {row['Weight']:.6f} (abs: {row['Absolute_Importance']:.6f})")
    
    # Plot numerical feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(num_importance_df.head(top_n)['Feature'], num_importance_df.head(top_n)['Absolute_Importance'])
    plt.xlabel('Absolute Weight')
    plt.ylabel('Feature')
    plt.title(f'Top Numerical Feature Importance in Ridge Regression')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('ridge_numerical_feature_importance.png')
    print(f"Numerical feature importance plot saved to ridge_numerical_feature_importance.png")
    
    return importance_df

def main():
    # Load housing data
    train_path = '/root/Rajnandini/test code/train_housingprices.csv'
    X_train, y_train, feature_names = load_housing_data(train_path)
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model with the best alpha found in the original script (0.1)
    best_alpha = 0.1
    print(f"Training Ridge Regression model with alpha={best_alpha}...")
    ridge = RidgeRegression(alpha=best_alpha, max_iter=5000, learning_rate=0.01)
    ridge.fit(X_train_scaled, y_train)
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(ridge, feature_names, top_n=20)
    
    # Print summary
    print("\n===== SUMMARY =====")
    print(f"Total features: {len(feature_names)}")
    print(f"Non-zero weights: {np.sum(np.abs(ridge.weights) > 1e-10)}")
    print(f"Most dominant feature: {importance_df.iloc[0]['Feature']} with absolute weight {importance_df.iloc[0]['Absolute_Importance']:.6f}")

if __name__ == "__main__":
    main()
