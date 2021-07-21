Linear Regression From Scratch
=======
Project Goal
---------------

The goal of this project was to create a basic linear regression model from scratch to understand the model in depth.

Steps and python folders 
-----------
  #### Linear_Regression_From_Scratch.py
    
    * Step 1: Load Data
      - Load in the dataset you want to try to model.
    
    * Step 2: Split The Data
      - Split the data into Features (X) and Labels (y).
      - X will take the shape M*N with M being the number of features your are using and N being the number of examples you have.
      - y will take the shape of 1xN with N being the number of examples you have. 
    
    * Step 3: Split the data again into Train, Validation and Test sets.
      - First split your data into Train and Test with 80% being the train data and 20% being the test data.
      - Then split your Train data again into Train and Validation sets with the same ratio as before (80/20).
    
    * Step 4: Normalisation or Standardisation
      - First calculate the Mean and the Standard Deviation (STD) of your Train dataset (Do Not Include Validation or Test Data!!!).
      - To calculate the Mean it's: sum(X)/Len(X).
      - To calculate the STD its: (Sum((X-Mean(x))^2)/len(X))^0.5.
      - To calculate the Normalisation it's: (X-Xmin)/(Xmax-Xmin) all values will fall between 0 and 1 after Normalisation.
      - To calculate the Standardisation it's: (X-Mean(X))/STD(X).
    
    * Step 5: Batch your X and Y
      - Batches are recommended if you have large datasets the size of the batch is usually a power of 2 e.g. (2,4,8,16,32...).
    
    * Step 6: Initalising a random Weight (W) and Bias (b)
      - A random Weight and Bias is initialised setting up the model which will use gradient descent to improve these two values.
      - W is a Vector with shape Nx1 with N being the number of features.
      - b is a Scalar.

    * Step 7: Predict Ŷ using X
      - Work out your predicted y.
      - This is done using the equation: Ŷ = WX + b.

    * Step 8: Calculate the gradient of W and b
      - To calculate the gradient of W it's: W' = 2*mean((Ŷ-y)^2)@X
      - To calculate the gradient of b it's: b' = 2*mean((Ŷ-y)^2)
    
    * Step 9: Set a Learning Rate (lr) (More on this in Optimations of Linear Regression)
      - The Learning Rate is a scalar that affects how much the Weight and Bias gradient affects the improved Weight and Bias. 

    * Step 10: Calculating The Mean Squared Error (Loss)
      - The Mean Squared Error allow us to evaluate how the model is performing
      - To calculate the Loss: Loss = mean((Ŷ-y)^2)
      - Typically a loss per batch is calculated and then averaged over all batches
    
    * Step 11: Calculating the Improved Weight and bias
      - Now that the Gradient of W and b are calulated, W and b can be improved
      - To improve W: W = W - lr*W'
      - To improve W: b = b - lr*b'
    
    * Step 12: Repetition
      - Repeat from Step 7 to Step 11 either for a set number of iterations or until the loss is below a satifactory level. 






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
