import joblib

model = joblib.load("models/delivery_time_model.pkl")

print("Model Type:")
print(type(model))

print("\nWeights:")
print(model.coef_)

print("\nIntercept:")
print(model.intercept_)

print("\nFeatures:")
print(model.feature_names_in_)

x = [5, 3, 1, 2]

manual = sum(w * v for w, v in zip(model.coef_, x)) + model.intercept_
sklearn_pred = model.predict([x])[0]

print(f"\nManual calc: {manual:.1f}")
print(f"sklearn pred: {sklearn_pred:.1f}")