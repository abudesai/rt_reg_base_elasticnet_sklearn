#Import required libraries
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 

from sklearn.linear_model import ElasticNet 
from sklearn.metrics import mean_squared_error


model_params_fname = "model_params.save"
model_fname = "model.save"
history_fname = "history.json"
MODEL_NAME = "ElasticNetSklearn"


class ElasticNetSklearn():     
    def __init__(self, l1_ratio=0.1, alpha=1, **kwargs) -> None:
        self.l1_ratio = np.float(l1_ratio)
        self.alpha = np.float(alpha)
        
        self.model = self.build_model()
        
        
        
    def build_model(self): 
        model = ElasticNet(l1_ratio= self.l1_ratio, alpha= self.alpha, random_state=0)
        return model
    
    
    def fit(self, train_X, train_y):        
                 
    
        self.model.fit(
                X = train_X,
                y = train_y
            )
    
    
    def predict(self, X, verbose=False): 
        preds = self.model.predict(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            # return self.model.score(x_test, y_test)        
            preds = self.model.predict(x_test)
            mse = mean_squared_error(y_test, preds, squared=False)
            return mse  

    
    def save(self, model_path): 
        joblib.dump(self.model, os.path.join(model_path, model_fname))


    @classmethod
    def load(cls, model_path): 
        elasticnet = joblib.load(os.path.join(model_path, model_fname))
        return elasticnet


def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path): 
    try: 
        model = ElasticNetSklearn.load(model_path)        
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


