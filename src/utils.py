# Calling mongo client , saving model things that will be used for all the project

import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score # statistical measure in a regression model that determines the proportion of variance in the dependent variable that can be explained by the independent variable

from src.exception import Custom_exception
import dill
def save_object(file_path,obj):
  try:
    dir_path = os.path.dirname(file_path)
    
    os.makedirs(dir_path,exist_ok = True)
    with open(file_path,"wb") as file_obj:
      dill.dump(obj,file_obj)
  except Exception as ex:
    raise(Custom_exception(ex,sys))

def evaluate_model(x_train,y_train,x_test,y_test,models):
  
  try:
    report = {}
    
    for i in range(len(models)):
      model = list(models.values())[i]
      
      model.fit(x_train,y_train)
      y_train_pred = model.predict(x_train)
      
      y_test_pred = model.predict(x_test)
      
      # train_model_score = r2_score(y_train,y_train_pred)
      
      test_model_score = r2_score(y_test,y_test_pred)
      
      report[list(models.keys())[i]] = test_model_score
    
    return report
  except Exception as ex:
    raise(Custom_exception(ex,sys))
    
