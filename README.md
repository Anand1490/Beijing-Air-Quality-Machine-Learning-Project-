# Beijing-Air-Quality-Machine-Learning-Project-
## Executive Summary:  
Using python, I took twelve different data sets from different stations around Beijing and merged them in order to use machine learning algorithms to figure out some factors that are causing certain air pollutants to be prevalent in the air as well as conducted some algorithm analysis to see which machine learning algorithm would perform the best for this project.  

1. Analyzed various factors contributing to the prevalence of several air pollutants.
2. Looked at performance of various machine learning algorithms  
3. Made recommendations of which machine learning algorithms to use as well as conclusions about what leads to air pollutants prevalance.   

## Business Problem:
 Beijing, China has had one of the worst air pollution in the world and one has to wonder what factors are causing these air pollutants to prevail in the air for so long. Knowing these factors can aid in solutions in creating cleaner air in the city. I also had to to look into which machine learning algorithms would be the best fit in using to solve this problem in future work on the project.   

## Methodology:
1. Conducted data cleaning, preprocessing, and feature engineering by first combining all the twelve data sets together into one, removing all the missin data, and standardizing all the data based on the mean of each column. Then finding the most important variables to use for further analysis. 
2. Ran a regression model in order to look at the prevalence of the air pollutant NO2 with variables such as Temperature, Dew point, and Wind Speed.
3. Ran a classification analysis to see which classifiers performed the best in relation to the project. 
4. Conducted clustering and association for further analysis on air pollutant NO2.

## Skills:
1.Python: Numpy, Scikit Learn, Matplotlib, Seaborn, Pandas, Statistics  
2.Data Cleaning   
3.Exploratory Data Analysis   
4.Visualization  
5.Modeling  
6.Feature Engineering  

## Results:
 Through feature engineering and running regression analysis, I found that the NO2 air pollutant is most prevalent  in colder temperatures and in particular possibly in the colder months and seasons. This could be attributed to increased gas from homes due to the increased heating usage in the colder temperatures. Streamlining these uses could decrease the air pollutant NO2 in the city. Some of the best classifiers that performed the best as shown in the figure below are Neural Networks and Logistic Regression which should be looked at to be used in future work. Finally, in the clustering and association phase of the project, found that there is low confidence in the weather being warm when there is high levels of NO2 which matches may conclusions from the regression phase of the project.   
![Alt Text]([(https://github.com/Anand1490/Beijing-Air-Quality-Machine-Learning-Project-/blob/main/algorithms.PNG]))   

## Next Steps:
1. Conduct time based analysis with the time features from the data set.
2. Use Neural Networks and Logistic Regression classifiers in future work. 
 
