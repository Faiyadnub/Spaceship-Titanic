# SpaceShip Titanic

## Introduction

The Spaceship Titanic was an interstellar passenger liner launched a month ago. 
With almost 13,000 passengers on board, the vessel set out on its maiden voyage 
transporting emigrants from our solar system to three newly habitable exoplanets 
orbiting nearby stars.

The aim of our work is to predict which passengers are transported to an alternate dimension, 
I will try to find main points which contributes to the risk of being transported to the 
alternative dimension while traveling with Spaceship Titanic.

## Project is divided into separate parts:

<li>Introduction</li>
<li>Notebook Preparation</li>
<li>Data Cleaning</li>
<li>Feature engineering</li>
<li>Exploratory Data Analysis</li>
<li>Missing Values - Feature Engineering</li>
<li>Modelling</li>
<li>Final Model</li>
<li>AutoML Model</li>
<li>Summary</li>
<li>Suggestion For Improvement</li>

## Project summary
<li>PassengerId split into GroupId and GroupSize.</li>
<li>VIP column was removed as it not giving us any useful information.</li>
<li>Filled missing values for both train and test datasets using various methods based on EDA analysis.</li>
<li>CryoSleep and SpentMoney show medium correlation with the target.</li>
<li>Numeric features show medium to low correlation.</li>
<li>Data is highly balanced.</li>
<li>One Home Planet per Family.</li>
<li>Most cabins stay at Decks E, F, and G, while cabin T shows an outlier.</li>
<li>Distribution over both decks is similar for Home Planet and Destination.</li>
<li>All people in groups stay on one side of the ship.</li>
<li>About ~91% of the cabins have only one destination planet.</li>
<li>35.8% of people on the ship are in cryo sleep, 64.2% are awake.</li>
<li>Filled missing values of Cabin Number using a linear regression model.</li>
<li>High outliers with spending features, capped maximum values at the 0.95 quantile.</li>
<li>People with age 12 and lower didn't spend money at all.</li>
<li>Most outliers stay with older people, who also spend the most.</li>
<li>The age distribution is skewed to the right, indicating a longer tail on the right side.</li>
<li>Based on overlapping distribution, it looks like younger people get transported more than older.</li>
<li>Additional bins created for age groups.</li>
<li>All groups have only one unique Home planet.</li>
<li>Reject the null hypothesis: There is a significant association between spaceship side and transported status.</li>
<li>Created ML Dummy Classifier model for baseline.</li>
<li>Used Boruta to check feature importance.</li>
<li>Created multiple ML models while evaluating which is the best based on F1 score and accuracy.</li>
<li>Used the best model to tune it a little bit more, evaluate it using the ROC curve, and predict results.</li>
<li>Created AutoML model, ran it for 10 minutes, to check its performance.</li>
<li>Submitted the best predictions to Kaggle competition - with the best accuracy of 0.80313.</li>


## Requirements for the project
To install all necessary libraries use - `pip install -r requirements.txt`

## Launch ML model locally
Install Docker and Java

Build Docker image -
`docker build -t spaceship-titanic-app .`

Run container -
`docker run -p 5000:5000 spaceship-titanic-app`

Main page -
`http://localhost:5000`

For predictions run -
`python predict.py`

Predictions will be saved in save folder where test.csv was placed -
`data/api_predictions.csv`

To check running containers -
`docker ps`

To stop docker container -
`docker stop 'container_id'`

## SpaceShip Titanic Dataset 
Dataset can be downloaded from 
[Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/data).


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact Information
[Email](ricardas.poskrebysev@gmail.com)
[LinkedIn](https://www.linkedin.com/in/ri%C4%8Dardas-poskreby%C5%A1evas-665207206/)
[GitHub](https://github.com/Riciokzz)
