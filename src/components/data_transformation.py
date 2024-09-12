# For cleaning and transforming data feature engineering and converting categorical to numerical

import sys,os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import  ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import Custom_exception
from src.logger import logging
from src.utils import save_object


@dataclass # gives default feature of __init__ , __rep__ and __str__
class DataTransformationConfig:
  preprocessor_obj_path = os.path.join('artifact','preprocessor.pkl')
  
class DataTransformation:
  def __init__(self):
    self.data_transformation_config = DataTransformationConfig()
  
  
  def get_data_transformer_obj(self): # creating pkl file for converting categ to num
    """
    implementing series of steps using pipeline
    """
    try:
      numerical_features = ['writing_score','reading_score']
      categorical_features = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
      
      num_pipeline = Pipeline(
        steps=[
          ("imputer",SimpleImputer(strategy = "median")), # median used for outlier robustness
          ("scalar",StandardScaler(with_mean=False))
        ]
      )
      cat_pipline = Pipeline(
        steps=[
          ("imputer",SimpleImputer(strategy="most_frequent")),
          ("one_hot_encoder",OneHotEncoder()),
          ("scalar",StandardScaler(with_mean=False))
          
        ]
      )
      logging.info(f"categorical columns encoded: {categorical_features}")
      logging.info(f"Numerical columns standard scaling completed nice :{numerical_features}")
      
      preprocessor = ColumnTransformer([
        ("num_pipeline",num_pipeline,numerical_features),
        ("cat_pipeline",cat_pipline,categorical_features)
      ])
      
      return preprocessor
    except Exception as e:
      raise(Custom_exception(e,sys))
    

  def initiate_data_transformation(self, train_path, test_path):
    try:
        # Read the train and test CSV files into DataFrames
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logging.info("Read train and test data completed")

        logging.info("Obtaining preprocessing object")
        preprocessing_obj = self.get_data_transformer_obj()

        target_col = "math_score"
        numerical_col = ["writing_score", "reading_score"]

        # Check if target column exists in the train and test sets
        if target_col not in train_df.columns:
            raise Custom_exception(f"{target_col} not found in training data", sys)
        if target_col not in test_df.columns:
            raise Custom_exception(f"{target_col} not found in test data", sys)

        # Dropping the target column from train and test sets
        input_feature_train = train_df.drop(columns=[target_col], axis=1)
        target_feature_train = train_df[target_col]

        input_feature_test = test_df.drop(columns=[target_col], axis=1)
        target_feature_test = test_df[target_col]

        # Print the dataframes before preprocessing for debugging
        logging.info(f"Input feature train columns: {input_feature_train.columns}")
        logging.info(f"Input feature test columns: {input_feature_test.columns}")

        logging.info("Applying preprocessing object on train and test data")

        # Apply preprocessing
        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train)
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test)

        # Combine the input features and target features
        train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train)]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test)]

        # Save the preprocessing object
        save_object(
            file_path=self.data_transformation_config.preprocessor_obj_path,
            obj=preprocessing_obj
        )

        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_path
        )

    except Exception as e:
        raise Custom_exception(e, sys)
