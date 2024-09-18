import sys
import pandas as pd
from src.exception import Custom_exception
from src.utils import load_object

class PredictPipeline:
  def __init__(self):
    pass
  
  def predict(self,features):
    try:
      model_path = 'artifact/model.pkl'
      preprocessor_path = 'artifact/preprocessor.pkl'
      model = load_object(file_path = model_path)
      preprocessor = load_object(file_path = preprocessor_path)
      data_scaled = preprocessor.transform(features)
      pred = model.predict(data_scaled)
      return pred
    except Exception as ex:
      raise Custom_exception(ex,sys)

class CustomData:
  def __init__(self,gender:str,race_ethnicity:str,parental_level_of_education:str,lunch:str,test_preparation_course:str,reading_score:int,writing_score:int):
    self.gender = gender
    self.race_ethnicity = race_ethnicity
    self.parental_level_of_education = parental_level_of_education
    self.test_preparation_course = test_preparation_course
    self.reading_score = reading_score
    self.writing_score = writing_score
    self.lunch = lunch

  def get_data_as_dataframe(self):
    try:
      custom_data_dict = {
        "gender" :[self.gender],
        "race_ethnicity":[self.race_ethnicity],
        "parental_level_of_education" : [self.parental_level_of_education],
        "lunch" : [self.lunch],
        "test_preparation_course" :[self.test_preparation_course],
        "reading_score" :[self.reading_score],
        "writing_score" :[self.writing_score]
      }
      df = pd.DataFrame(custom_data_dict)
      # print(df)
      return df
    except Exception as ex:
      raise Custom_exception(ex,sys)
    
    