import json
from datetime import datetime
import numpy as np
import sklearn
import sys
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
from pprint import pprint
from joblib import dump
import os

from Tesina_General_Utils import make_dir 

def save_model(model, base_filename, output_path, min_vals=None, max_vals=None, label_classes=None):
    """
    Saves a scikit-learn model to a file.

    :param model: Trained model to be saved.
    :param base_filename: Base filename for saving the model.
    :param output_path: Directory path where the model file will be saved.
    """
    # Ensure the output directory exists
    model_dir = make_dir([output_path])
    model_filename = os.path.join(model_dir, f"{base_filename}_model.joblib")

    model_data = {
        'model': model,
        'min_vals': min_vals,
        'max_vals': max_vals,
        'label_classes': label_classes
    }
    # Save the model
    dump(model_data, model_filename)
    print(f"Model saved to: {model_filename}")
    
def save_model_metadata_and_analysis(model, X_train, y_train, X_test, y_test, feature_names, base_filename, output_path, print_info=False):
    """
    Saves metadata and quality analysis of a model to a JSON file.

    :param model: Trained model.
    :param X_train: Training data.
    :param y_train: Training labels.
    :param X_test: Testing data.
    :param y_test: Testing labels.
    :param feature_names: Names of the features.
    :param base_filename: Base filename for saving the metadata and analysis.
    :param output_path: Directory path where the metadata file will be saved.
    """

    
    metadata = {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': type(model).__name__,
        'model_parameters': model.get_params(),
        'python_version': sys.version,
        'sklearn_version': sklearn.__version__,
        'feature_names': feature_names,
        'class_labels': np.unique(y_test).tolist(),
        'dataset_size': {'train': len(X_train), 'test': len(X_test)}
    }

    # Model analysis
    metadata['training_score'] = model.score(X_train, y_train)
    metadata['test_score'] = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    metadata['classification_report'] = classification_report(y_test, predictions, output_dict=True)

    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)

    try:
        # AUC and ROC curve (if applicable)
        if hasattr(model, "predict_proba"):
            proba_predictions = model.predict_proba(X_test)
            metadata['roc_auc_score'] = roc_auc_score(y_test, proba_predictions, multi_class='ovr')
    except ValueError:
        # Can't compute ROC AUC score for multiclass classification with 'ovr' scheme
        pass
    
    # Feature importances (if applicable)
    if hasattr(model, "feature_importances_"):
        metadata['feature_importances'] = dict(zip(feature_names, model.feature_importances_))

    # Cross-validation scores
    # Aquí puedes ajustar los parámetros de cv y scoring según tus necesidades
    metadata['cross_validation_scores'] = cross_val_score(model, X_train, y_train, cv=5).tolist()
    
    # Print information if required
    if print_info:
        pprint(metadata)

    if base_filename:
        metadata_dir = make_dir([output_path])
        metadata_filename = os.path.join(metadata_dir, f"{base_filename}_metadata.json")
        with open(metadata_filename, 'w') as file:
            json.dump(metadata, file, indent=4)
        print(f"Metadata and model analysis saved to: {metadata_filename}")
        
        