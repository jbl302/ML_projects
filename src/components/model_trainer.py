# For using models and evaluating them

import sys,os
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (GradientBoostingRegressor,AdaBoostRegressor,RandomForestRegressor)

from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import Custom_exception
from src.logger import logging
from src.utils import evaluate_model,save_object



@dataclass
class ModelTrainerConfig:
  train_model_path = os.path.join('artifact','model.pkl')
  
class ModelTrainer:
  def __init__(self) :
    self.model_trainer_config = ModelTrainerConfig()
  
  def inititate_model_trainer(self,train_arr,test_arr):
    
    try:
      logging.info("Splitting traning and test input data")
      x_train,y_train,x_test,y_test = (
      train_arr[:,:-1] ,
      train_arr[:,-1],
      test_arr[:,:-1],
      test_arr[:,-1]
                  )
      
      models = {
        "Random_forest":RandomForestRegressor(),
        "Decision_tree":DecisionTreeRegressor(),
        "Linear_regression":LinearRegression(),
        "knn":KNeighborsRegressor(),
        "xgboost":XGBRegressor(),
        "catboost":CatBoostRegressor(),
        "adaboost":AdaBoostRegressor(),
        "gradient_boost":GradientBoostingRegressor()
        
      }
      
      model_report :dict = evaluate_model(
        x_train = x_train,
        y_train = y_train,
        x_test = x_test,
        y_test = y_test,
        models = models)
      
      
      model_names = model_report.keys()
      model_scores = model_report.values()
      best_model_score = max(list(model_scores))
      
      best_model_name = list(model_names)[list(model_scores).index(best_model_score)]
      if best_model_score<.6:
        raise(Custom_exception("no best model"),sys)
      
      logging.info(f"Best model found : {best_model_name}")
      
      best_model = models[best_model_name]
      
      save_object(
        file_path=ModelTrainerConfig().train_model_path,
        obj=best_model) # save pickle file of model
      
      predict = best_model.predict(x_test)
      
      r2_square = r2_score(y_test,predict)
      
      return r2_square
      
      
    except Exception as ex:
      raise(Custom_exception(ex,sys))
    
    
    
