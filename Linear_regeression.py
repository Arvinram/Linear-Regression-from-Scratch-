
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 1. GENERATE HOUSE DATA
print("=" * 55)
print("LINEAR REGRESSION FROM SCRATCH")
print("=" * 55)

rng = np.random.default_rng(3)
n = 400

# Generate house size in square feet
size = np.clip(rng.normal(1800, 450, n), 500, 4000)

# Generate number of bedrooms
bedrooms = np.rint(
    1 + (size - 500) / 700 + rng.normal(0, 0.8, n)
).astype(int)
bedrooms = np.clip(bedrooms, 1, 6)

# Generate house age
age = rng.integers(0, 51, n)

# Calculate house prices with some random noise
price = (
    50000
    + 130 * size
    + 8000 * bedrooms
    - 1200 * age
    + rng.normal(0, 25000, n)
)

# 2. CREATE DATAFRAME USING PANDAS
df = pd.DataFrame({
    "Size_sqft": size,
    "Bedrooms": bedrooms,
    "Age_years": age,
    "Price": price
})

df["Size_sqft"] = df["Size_sqft"].round(2)
df["Price"] = df["Price"].round(2)

print("\nFIRST FIVE HOUSE RECORDS")
print(df.head())

print("\nDATASET INFORMATION")
print("Number of houses:", len(df))
print("Columns:", df.columns.tolist())

# Save dataset
df.to_csv("houses.csv", index=False)
print("\nhouses.csv created successfully!")

# 3. PREPARE FEATURES AND TARGET
X = df[["Size_sqft", "Bedrooms", "Age_years"]].to_numpy()
y = df["Price"].to_numpy()

# Split data: 80% training, 20% testing
indices = rng.permutation(n)
split = int(0.8 * n)

train_idx = indices[:split]
test_idx = indices[split:]

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print("\nTRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))

# 4. STANDARDISE FEATURES
x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0)
x_std[x_std == 0] = 1

y_mean = y_train.mean()
y_std = y_train.std()

X_train_s = (X_train - x_mean) / x_std
X_test_s = (X_test - x_mean) / x_std
y_train_s = (y_train - y_mean) / y_std

# Add intercept column
X_train_b = np.column_stack([
    np.ones(len(X_train_s)), X_train_s
])
X_test_b = np.column_stack([
    np.ones(len(X_test_s)), X_test_s
])

# 5. METHOD 1: NORMAL EQUATION
w_normal = np.linalg.pinv(X_train_b) @ y_train_s

# 6. METHOD 2: GRADIENT DESCENT
w_gd = np.zeros(X_train_b.shape[1])
learning_rate = 0.1
epochs = 500
loss_history = []

for epoch in range(epochs):
    error = X_train_b @ w_gd - y_train_s

    gradient = (2 / len(y_train_s)) * (
        X_train_b.T @ error
    )

    w_gd -= learning_rate * gradient

    loss = np.mean(
        (X_train_b @ w_gd - y_train_s) ** 2
    )
    loss_history.append(loss)

# 7. METHOD 3: NUMPY LEAST SQUARES
w_lstsq = np.linalg.lstsq(
    X_train_b, y_train_s, rcond=None
)[0]

# 8. EVALUATE THE MODELS
def evaluate(weights):
    predicted_standard = X_test_b @ weights

    # Convert predictions back to actual prices
    predictions = (
        predicted_standard * y_std + y_mean
    )

    residuals = y_test - predictions

    rmse = np.sqrt(np.mean(residuals ** 2))

    ss_res = np.sum(residuals ** 2)
    ss_total = np.sum(
        (y_test - y_test.mean()) ** 2
    )

    r2 = 1 - ss_res / ss_total

    return predictions, rmse, r2, residuals


models = {
    "Normal Equation": w_normal,
    "Gradient Descent": w_gd,
    "NumPy Least Squares": w_lstsq
}

results = {}

print("\nMODEL PERFORMANCE")
print("-" * 55)

for name, weights in models.items():
    predictions, rmse, r2, residuals = evaluate(weights)

    results[name] = {
        "predictions": predictions,
        "rmse": rmse,
        "r2": r2,
        "residuals": residuals
    }

    print("\nModel:", name)
    print("RMSE: $", round(rmse, 2))
    print("R-squared:", round(r2, 4))

# 9. COMPARE MODEL COEFFICIENTS
coefficient_table = pd.DataFrame(
    {name: weights for name, weights in models.items()},
    index=["Intercept", "Size", "Bedrooms", "Age"]
)

print("\nSTANDARDISED MODEL COEFFICIENTS")
print(coefficient_table.round(4))

# 10. PREDICT A NEW HOUSE PRICE
new_house = np.array([[1800, 3, 10]])

new_house_s = (new_house - x_mean) / x_std

new_house_b = np.column_stack([
    np.ones(len(new_house_s)), new_house_s
])

predicted_price = (
    new_house_b @ w_lstsq
) * y_std + y_mean

print("\nNEW HOUSE PRICE PREDICTION")
print("House size: 1800 sq. ft.")
print("Bedrooms: 3")
print("House age: 10 years")
print(f"Predicted price: ${predicted_price[0]:,.2f}")

# 11. CREATE THREE CHARTS
best = results["NumPy Least Squares"]
predictions = best["predictions"]
residuals = best["residuals"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Chart 1: Actual and predicted prices
axes[0].scatter(
    X_test[:, 0],
    y_test,
    alpha=0.7,
    label="Actual prices"
)

order = np.argsort(X_test[:, 0])

axes[0].plot(
    X_test[order, 0],
    predictions[order],
    label="Predicted prices"
)

axes[0].set_title("House Price vs Size")
axes[0].set_xlabel("Size (sq. ft.)")
axes[0].set_ylabel("Price ($)")
axes[0].legend()

# Chart 2: Gradient descent loss
axes[1].plot(loss_history)
axes[1].set_yscale("log")
axes[1].set_title("Gradient Descent Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Mean Squared Error")

# Chart 3: Prediction residuals
axes[2].scatter(
    predictions,
    residuals,
    alpha=0.7
)

axes[2].axhline(0, linestyle="--")
axes[2].set_title("Test Residuals")
axes[2].set_xlabel("Predicted Price ($)")
axes[2].set_ylabel("Actual - Predicted ($)")

plt.tight_layout()

# Save chart
plt.savefig("chart.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()

print("\nchart.png created successfully!")

# 12. FINAL SUMMARY
print("\n" + "=" * 55)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 55)
print("Dataset file: houses.csv")
print("Chart file: chart.png")
print("Total houses:", len(df))
print("Training houses:", len(X_train))
print("Testing houses:", len(X_test))
print("Best model by test R-squared: NumPy Least Squares")
print("R-squared:", round(results["NumPy Least Squares"]["r2"], 4))
print("RMSE: $", round(results["NumPy Least Squares"]["rmse"], 2))
