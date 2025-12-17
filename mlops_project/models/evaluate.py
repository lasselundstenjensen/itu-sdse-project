
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": conf_matrix
    }



def compare_models(results_dict):
    """Compare models and choose."""
    best_model = None
    best_f1 = 0
    
    for model_name, results in results_dict.items():
        f1 = results["report"]["weighted avg"]["f1-score"]
        print(f"{model_name}: F1 = {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model_name
    
    print(f"\nBest model: {best_model}")
    return best_model