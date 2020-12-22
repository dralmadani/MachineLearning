<h1>Predicting the Onset of cardiovascular disease</h1>
<h2>by Chaokai Zhang, Abdulsalam Almadani, Winnie Mkandawire, Trusting Inekwe</h2>

![heartdiseasepic](https://user-images.githubusercontent.com/55979883/101299304-bd79db00-37ff-11eb-837f-2b42b7b4d399.jpeg)

<p align="justify">Cardiovascular disease is the leading cause of death among people in the US. According to the National Vital Statistics, about 854,390 people died due to Cardiovascular Disease in 2019 only (CDC, 2019). Despite this high mortality rate, health care service in the US is expensive and so providing a means in which people could detect (easily and possibly early) if they have a Cardiovascular disease or not would help in early treatment, lesser medical expenses and would increase chances of survival. Our proposed Cardiovascular Disease Onset Prediction tool would be used by medical personnel to detect Cardiovascular disease easily and faster. The current system only allows qualified personnel to collate patients’ clinical information and make decision – which usually take a longer time – on whether a patient has been diagnosed with Cardiovascular disease or not.</p>

<p align="justify">We design a Cardiovascular Disease onset prediction​ tool that will help in predicting onset of Cardiovascular disease given a patients’ clinical information. </p>

<h3>Our Approach</h3>
<h4>Step 1: Data Extraction:</h4>
<p align="justify">Our dataset is from the Kaggle https://www.kaggle.com/sulianova/cardiovascular-disease-dataset (Ulianova, n.d.). it has 70,000 records of patients’ data with 11 features –both numerical and categorical.</p>

<h4> Step 2: Data Pre-processing: </h4>
<p align="justify"> Our data processing involved three steps namely:</p>

<h5>i. Checking for missing or NaN values and removal of unimportant features</h5>
<p align="justify">As a first step in our data processing, we removed the id column because it didn't give us any information. We also looked for Nan values of which we didn't find any. We had a dataset that was complete and numeric</p>

<h5>ii. Checking if our dataset is balanced</h5>
<p align="justify">Secondly, we checked to see if our dataset was balanced. Our dataset had a Positive to Negative ratio of 34979 to 35021 indicating an almost balanced dataset. </p>
  
<h5>iii. Removing outliers</h5>
<p align="justify"> Our dataset contained outliers for some of the features. For example, for the various blood pressure categories of Normal, Elevated and Hypertensive stages [1], we searched for values that didnt fall within the acceptable  Systolic (upper) and Diastolic (lower) ranges. For our Systolic Blood Pressure feature, we set our outliers values to those that were less than or equal to 80 or greater than 200 of which we had over 307 cases. (From left to right: raw data vs data after outlier removal)</p>

![systolic](https://user-images.githubusercontent.com/55979883/101568705-146ae600-39a1-11eb-8bd0-884cbcd64e17.jpeg)

<p align="justify">For Diastolic Blood Pressure feature, we set our outliers values to those that were less than 50 or greater than 1032 of had over 1032 cases. (From left to right: raw data vs data after outlier removal)</p>

![Diastollic](https://user-images.githubusercontent.com/55979883/101568587-d1a90e00-39a0-11eb-8a7c-8629b82ed449.jpeg)

<p align="justify">For weight feature, we set our outliers values to those that were less than or equal to 30 of had over 7 cases. (From left to right: raw data vs data after outlier removal)</p>

![Weight](https://user-images.githubusercontent.com/55979883/101305205-22d5c800-3810-11eb-88b0-46836c651faa.png)

<p align="justify">For height feature, we set our outliers values to those that were less than or equal to 100 of had over 29 cases. (From left to right: raw data vs data after outlier removal)</p>

![Height](https://user-images.githubusercontent.com/55979883/101305046-b22eab80-380f-11eb-81d7-b81f6ca0580e.png)

<h4>Step 3: created a heat map to find the correlation of features</h4>
<p align="justify">From the Heat map below, shows the features age, ap_hi, ap_lo and cholesterol as having correlation with our target feature which is cardio</p>

![corr_ML](https://user-images.githubusercontent.com/55979883/101564433-2d6f9900-3999-11eb-9f5f-e26e338609c0.png)

<h4>Step 4: Selecting important features</h4>
<p align="justify">We found out features of importance and selected the first 4 important features; age(0), ap_hi(4), ap_lo(5) and cholesterol(6) as important features because they had significant correlating values and we created our models using these four features</p>

![Feature Importance](https://user-images.githubusercontent.com/55979883/101569181-f9e53c80-39a1-11eb-9977-7d7b229f67c7.jpeg)

<h4>Step 5: Modelling using Machine Learning Algorithms</h4>
<p align="justify">We used Machine learning algorithms including Naive Bayes, XGBoost, KNN, Decision tree, random forest classifier, Support Vector Classifier, Perceptron as well as Convolutional Neural network for prediction. We used optimizer methods for the models by tuning them with different hyperparameter measures for performance improvement. We also apply 10-fold cross-validation to each of the models to ensure the models perform the best result.</p>

<table style="width:100%">
  <caption>Model Algorithms implemented</caption>
  <tr>
    <td>K Nearest Neighbour</td>
    <td>Logistic Regression</td>
    <td>Decision Tree Classifier</td>
    <td>Random Forest Classifier</td>
    <td>Extra Tree Classifier</td>
    <td>Support Vector Classifier</td>
    <td>Naive Bayes</td>
  </tr>
  <tr>
    <td>Multinomial Naive Bayes</td>
    <td>Bagging Classifier</td>
    <td>AdaBoost Classifier</td>
    <td>XGBoost Classifier</td>
    <td>Neural Networks</td>  
  </tr>
</table>

<p align="justify">The figure below is a list of models we used and their accuracy values</p>

![model_performance2](https://user-images.githubusercontent.com/55979883/101528076-cd5b0180-395c-11eb-9b49-12a6cbb1fdee.png)

<img width="1089" alt="ACCURACY SCORES" src="https://user-images.githubusercontent.com/55979883/101582673-06b95e80-39a9-11eb-8e65-6ea81a910577.png">

<p align="justify">The figure below is the AUC ROC for models we used</p>

![AUC ROC ](https://user-images.githubusercontent.com/55979883/101567442-d076e180-399e-11eb-9cb8-d8ba9c623e41.jpeg)

<!--![prediction_result](https://user-images.githubusercontent.com/55979883/101528002-b9170480-395c-11eb-83cd-0fbd58bc048d.png) -->


<h4>Step 6: Hybrid Algorithm</h4>
<p align="justify">Finally, we selected some of our best models namely; Support Vector Classifier, Logistic Regression and Adaboost(excluding XGBoost because of high computational cost) to make a hybrid model so we can have a better accuracy. Our hybrid model uses a voting classifier to select the most common prediction as the best prediction. Our hybrid model gave us an accuracy of 71.93%.</p>

![funnel](https://user-images.githubusercontent.com/55979883/101667351-3d30c100-3a1d-11eb-9de7-decec63427e9.jpeg)

<h4>Step 6: Conclusion</h4>
<p align="justify">To conclude, we designed a tool to predict the onset of cardiovascular disease. To create this tool, we tried out a couple of models to find out which model gave us the best prediction after training our models with our dataset. We selected three models namely; AdaBoost, SVC and Logistic Reggression and created a hybrid model to see if we can get even better predictions. From our hybrid model, we can see that there isn't much difference in accuracy values between the hybrid model and the some of our selected models but we have more confidence in our hybrid model because we use a voting classifier of the three selected models.</p>


<h3>Online Source</h3>
<p> 1. https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings </p>
