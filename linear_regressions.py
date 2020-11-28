from Cleaning_data import clean_data
import numpy as np
from sklearn.linear_model import LinearRegression


processed_data = clean_data.PreprocessedData()
patient_data = processed_data.get_patient_data()
tappy_data = processed_data.get_tappy_data()



def parkinsons_and_mean_hold_time():

    x = []
    y = []

    for i in tappy_data:
        if i[0] in patient_data:
            if patient_data[i[0]]['Parkinsons:'] == True:
                y.append(1)
            else:
                y.append(0)
        
            x.append(tappy_data[i]['mean_flight_time'])
    print(x)
    print(y)



def find_linear_regressions():
    



# Calculate the mean squared error to determine the effectivness of
# a given model



            
            




