import joblib
import sys
from pathlib import Path

model_path = Path('data/models/ensemble_model_calibrated_isotonic.joblib')

if not model_path.exists():
    print(f"Model not found at {model_path}")
    sys.exit(1)

try:
    model = joblib.load(model_path)
    print(f"Type: {type(model).__name__}")
    
    # Check for underlying estimator
    base_model = None
    if hasattr(model, 'estimator'):
        print(f"Has 'estimator' attribute: {type(model.estimator).__name__}")
        base_model = model.estimator
    elif hasattr(model, 'base_estimator'):
        print(f"Has 'base_estimator' attribute: {type(model.base_estimator).__name__}")
        base_model = model.base_estimator
        
    if base_model:
        if hasattr(base_model, 'estimators_'):
            print(f"Ensemble Estimators ({len(base_model.estimators_)}):")
            for i, est in enumerate(base_model.estimators_):
                print(f"  {i}: {type(est).__name__}")
        elif hasattr(base_model, 'estimators'): # StackingClassifier before fit
             print(f"Ensemble Estimators (List): {len(base_model.estimators)}")
             
    # Check calibrated classifiers
    if hasattr(model, 'calibrated_classifiers_'):
        print(f"Calibrated Classifiers: {len(model.calibrated_classifiers_)}")
        if len(model.calibrated_classifiers_) > 0:
            first_cc = model.calibrated_classifiers_[0]
            print(f"First CC Base Estimator: {type(first_cc.base_estimator).__name__}")

except Exception as e:
    print(f"Error loading model: {e}")
