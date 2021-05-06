#Loading dataframe 
import pandas as pd
columns = ['Year', 'No smoothing', 'Lowest', 'Africa', 'Asia', 'Europe' ,'North America', 'South America', 'Oceania']
df = pd.read_csv('Main_dataset_globalwarming.csv', names = columns)
#print (df.tail())

#Normalizing data by using min-max
from sklearn.preprocessing import MinMaxScaler
pre_scaled = pd.DataFrame()
pre_scaled['Year'], pre_scaled['No smoothing'] = df['Year'], df['No smoothing']
min_max_scaler = MinMaxScaler()
data_minmax = min_max_scaler.fit_transform(pre_scaled)
scaled_df = pd.DataFrame(data_minmax, columns = ['Year', 'No smoothing'])
#print(scaled_df.tail())


#Creating model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
X_train, X_test, Y_train, Y_test = train_test_split(scaled_df['Year'], scaled_df['No smoothing'])
#print(X_train.tail())
X_train_df, X_test_df = pd.DataFrame(X_train), pd.DataFrame(X_test)

degree_usr = 5
poly = PolynomialFeatures(degree = degree_usr)
X_train_poly, X_test_poly = poly.fit_transform(X_train_df), poly.fit_transform(X_test_df)

from sklearn import linear_model
model = linear_model.LinearRegression()
model = model.fit(X_train_poly, Y_train)
coefficient = model.coef_
intercep = model.intercept_

import matplotlib.pyplot as plt
import numpy as np 
step = 1/139
x_axis = np.arange(0, 1, step)

#Creating the polynomial equation
response =  intercep + coefficient[1] * x_axis + coefficient[2] * x_axis**2  + coefficient[3]*x_axis**3 + coefficient[4]*x_axis**4 + coefficient[5]*x_axis**5 #+ coefficient[6]*x_axis**6

#Metrics
from sklearn.metrics import r2_score
prediction = model.predict(X_test_poly)
r_squared = r2_score(prediction, Y_test)
print('Prediction: ', prediction)
print('\n')
print('R^2: ', r_squared)

rscld_df = min_max_scaler.inverse_transform(scaled_df)
rscld_df = pd.DataFrame(rscld_df, columns = ['Year', 'No smoothing'])
#print(rscld_df.tail())

rscld_res = pd.DataFrame(response, columns = ['No smoothing'])
rscld_axis = pd.DataFrame(x_axis, columns = ['Year'])
rscld_prediction = pd.DataFrame(prediction, columns = ['No smoothing'])
rscld_response = pd.DataFrame()
rscld_response['Year'], rscld_response['No smoothing'] = rscld_axis['Year'], rscld_res['No smoothing']

rscld_response = min_max_scaler.inverse_transform(rscld_response)
rscld_response = pd.DataFrame(rscld_response, columns = ['Year', 'No smoothing'])

#print(rscld_response.tail())

plt.scatter(rscld_df['Year'], rscld_df['No smoothing'], color = 'b')
plt.plot(rscld_response['Year'], rscld_response['No smoothing'], color = 'r')
plt.xlabel('Years')
plt.ylabel('Offset °C from ideal temperature')
plt.legend(['Model', 'Raw data'], loc = 'upper left')
plt.show()

choice = input('Would you like to make a prediction? (Y/N) \n')
while choice == 'Y':
    Target = input('Type year you wish to know temp of\n')
    Target = int(Target)  + 1
    
    size = Target - 1880
    ro3 = size/139
    stepp  = 1 / 139
    
    axis = np.arange(0, ro3, stepp)
    axis_fake = np.arange(1880, Target, 1)
    
    period = len(axis)
    array = np.reshape(axis, (period, 1))
    array_fake = np.reshape(axis_fake, (len(axis_fake), 1))

    res =   intercep + coefficient[1] * axis + coefficient[2] * axis**2  + coefficient[3]*axis**3 + coefficient[4]*axis**4 + coefficient[5]*axis**5 #+ coefficient[6]*axis**6
    df_res = pd.DataFrame(res, columns = ['Temperature'])
    df_axis = pd.DataFrame(array, columns = ['Year'])
    df_fake = pd.DataFrame(array_fake, columns = ['Year'])
    
    final_df = pd.DataFrame()
    final_df['Year'], final_df['Temperature'] = df_axis['Year'], df_res['Temperature']
    final_df = min_max_scaler.inverse_transform(final_df)
    final_df = pd.DataFrame(final_df, columns = ['Year', 'Temperature'])
    final_final =pd.DataFrame()
    final_final['Year'], final_final['Offset °C from ideal temp'] = df_fake['Year'], final_df['Temperature']
    print(final_final.tail())
    
    choice = input('Would you like to make another prediction? (Y/N)\n')

