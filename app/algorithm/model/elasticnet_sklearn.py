#Import required libraries
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 

from sklearn.linear_model import ElasticNet


model_params_fname = "model_params.save"
model_fname = "model.save"
history_fname = "history.json"
MODEL_NAME = "ElasticNet_sklearn"

class ElasticNet_sklearn(): 
    
    def __init__(self, l1_ratio=0.1, alpha=1, **kwargs) -> None:
        super(ElasticNet_sklearn, self).__init__(**kwargs)
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
            return self.model.score(x_test, y_test)        

    
    def save(self, model_path): 
        model_params = {
            "l1_ratio": self.l1_ratio,
            "alpha":self.alpha
            
        }
        joblib.dump(model_params, os.path.join(model_path, model_params_fname))

        joblib.dump(self.model, os.path.join(model_path, model_fname))

    @classmethod
    def load(cls, model_path): 
        model_params = joblib.load(os.path.join(model_path, model_params_fname))

        elasticnet = joblib.load(os.path.join(model_path, model_fname))
        return elasticnet


def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path): 
    try: 
        model = ElasticNet_sklearn.load(model_path)        
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


def save_training_history(history, f_path): 
    hist_df = pd.DataFrame(history.history) 
    hist_json_file = os.path.join(f_path, history_fname)
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)