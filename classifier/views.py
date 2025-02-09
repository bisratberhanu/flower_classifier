import os
import joblib
import pandas as pd
from django.shortcuts import render
from django.conf import settings

# Load the model and metadata once at app startup
model_path = os.path.join(settings.BASE_DIR, 'classifier', 'decision_tree_model.pkl')
model_data = joblib.load(model_path)
model = model_data["model"]
feature_names = model_data["feature_names"]
target_names = model_data["target_names"]

def classify_flower(request):
    predicted_class = None
    decision_path = None

    if request.method == "POST":
        # Extract input features from form data
        feature_values = request.POST.getlist('features')
        feature_values = [float(value) for value in feature_values]

        # Prepare data for prediction
        input_features = pd.DataFrame([feature_values], columns=feature_names)

        # Predict class
        prediction = model.predict(input_features)[0]
        predicted_class = target_names[prediction]

        # Generate decision path
        node_indicator = model.decision_path(input_features)
        leave_id = model.apply(input_features)

        decision_path = ""
        for node_id in node_indicator.indices:
            if leave_id[0] == node_id:
                decision_path += f"Reached leaf node {node_id}: Predict class {predicted_class}\n"
            else:
                if input_features.iloc[0, model.tree_.feature[node_id]] <= model.tree_.threshold[node_id]:
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"
                decision_path += (
                    f"Node {node_id}: (Feature '{feature_names[model.tree_.feature[node_id]]}' "
                    f"is {threshold_sign} {model.tree_.threshold[node_id]:.2f})\n"
                )

    # Render the page with results (if any)
    return render(request, 'index.html', {
        'feature_names': feature_names,
        'predicted_class': predicted_class,
        'decision_path': decision_path
    })
