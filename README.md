Linear Regression From Scratch
=======
Project Goal
---------------

The goal of this project was to create a basic linear regression model from scratch to understand the model in depth.

Steps and python folders 
-----------
Data scraping:

  #### Linear_Regression_From_Scratch.py
    
    * __Step 1:__ Load Data
      - Load in the dataset you want to try to model.
    
    * **Step 2:** Split The Data
      - Split the data into Features (X) and Labels (y).
      - X will take the shape M*N with M being the number of features your are using and N being the number of examples you have.
      - y will take the shape of 1xN with N being the number of examples you have. 
    
    * **Step 3:** Split the data again into Train, Validation and Test sets.
      - First split your data into Train and Test with 80% being the train data and 20% being the test data.
      - Then split your Train data again into Train and Validation sets with the same ratio as before.

Imported Libaries
-----------   

1. selenium import webdriver
2. selenium.webdriver.common import keys
3. selenium.webdriver.common.keys import Keys
4. time import sleep
5. datetime import timedelta, datetime
6. pprint import pprint
7. import pandas as pd
8. import json

Amount of Data Scraped
----------------------

* 560 cities have been scraped
* Each city contains 12 years of weather data
* 4,568 rows per city
* 11 columns per city
* A total of 2,558,080 rows x 11 columns
