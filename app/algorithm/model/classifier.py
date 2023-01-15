
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 

from sklearn.ensemble import RandomForestClassifier



model_fname = "model.save"
MODEL_NAME = "text_class_base_random_forest_sklearn"


class Classifier(): 
    
    def __init__(self, n_estimators = 200, min_samples_split = 5, min_samples_leaf = 3, **kwargs) -> None:
        self.n_estimators = int(n_estimators)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.model = self.build_model()     
        
        
    def build_model(self): 
        model = RandomForestClassifier(n_estimators = self.n_estimators, min_samples_split = self.min_samples_split, min_samples_leaf = self.min_samples_leaf)
        return model
    
    
    def fit(self, train_X, train_y):        
        self.model.fit(train_X, train_y)            
        
    
    def predict(self, X): 
        preds = self.model.predict(X)
        return preds 
    
    
    def predict_proba(self, X): 
        preds = self.model.predict_proba(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)           

    
    def save(self, model_path): 
        joblib.dump(self, os.path.join(model_path, model_fname))        


    @classmethod
    def load(cls, model_path):         
        model = joblib.load(os.path.join(model_path, model_fname))
        return model


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):     
    model = joblib.load(os.path.join(model_path, model_fname))   
    return model


