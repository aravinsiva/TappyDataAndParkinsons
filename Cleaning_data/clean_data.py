import os
import numpy 
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import *
from keras.layers import *
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import *
from sklearn.linear_model import LinearRegression

class PreprocessedData():

    def __init__(self):
        self.tappy_data = self.create_tappy_dict()
        self.patient_data = self.create_patient_dict()

    def get_tappy_data(self):
        return self.tappy_data
    
    def get_patient_data(self):
        return self.patient_data
        
    # Create a dictionry of all patient data
    def create_patient_dict(self):
        archived_users = os.listdir('../Archived users')
        all_patients_demographic_data = {}

        for i in archived_users:
            all_patients_demographic_data[i[5:15]] = self.clean_demographic_data(i)
        
        return all_patients_demographic_data

    # Create a dictionary of all tappy data
    def create_tappy_dict(self):
        all_patient_tappy_data = {}
        tappy_data = os.listdir('../Tappy Data')

        for i in tappy_data:
            all_patient_tappy_data[tuple([i[0:10],i[12:16]])] = self.clean_tappy_data(i)
        
        return all_patient_tappy_data

    def clean_demographic_data(self, file_name):
        first= open('../Archived users/' + file_name, 'r')
        lines = first.readlines()
        dictionary = {}
        for i in lines:
            value = i[i.index(':')+1:].strip(' ')
            key = i[:i.index(':')+1].strip(' ')

            if value[0] == '\n' or value[0] == '-':
                dictionary[key] = None

            else:
                dictionary[key] = value.strip('\n')

        return dictionary

    def clean_tappy_data(self, file_name):
        second= open('../Tappy Data/' + file_name)
        lines = second.readlines()

        dictionary = {}

        mean_flight_time = 0
        mean_latency_time = 0
        mean_hold_time = 0

        num_data_points = 0

        for i in lines:
            vals = i.split('\t')
            
            if (len(vals[4]) == 6 and len(vals[6]) == 6 and len(vals[7]) == 6):
                mean_flight_time += float(vals[4])
                mean_latency_time += float(vals[6])
                mean_hold_time += float(vals[7])
                num_data_points += 1
            
            

        if (num_data_points > 0):
            dictionary['mean_flight_time'] = mean_flight_time/num_data_points
            dictionary['mean_latency_time'] = mean_latency_time/num_data_points
            dictionary['mean_hold_time'] = mean_hold_time/num_data_points

        return dictionary




class CreateLinearRegressionModels():

    def __init__(self):
        self.processed_data = PreprocessedData()
        self.patient_data = self.processed_data.get_patient_data()
        self.tappy_data = self.processed_data.get_tappy_data()


    def create_single_regressions(self):
    # Create all possible combinations of single regressions
    # Find the most accurate of all of them using mean squared error
        x1 = []
        x2 = []
        x3 = []

        y1= []
        y2 = []
        y3 = []


        for i in self.tappy_data:

            if i[0] in self.patient_data and 'mean_flight_time' in self.tappy_data[i]: 
                if self.patient_data[i[0]]['Parkinsons:'] == 'True':
                    y1.append(1)
                else:
                    y1.append(0)
                
                x1.append(self.tappy_data[i]['mean_flight_time'])

            
            if i[0] in self.patient_data and 'mean_latency_time' in self.tappy_data[i]: 

                if self.patient_data[i[0]]['Parkinsons:'] == 'True':
                    y2.append(1)
                else:
                    y2.append(0)

                x2.append(self.tappy_data[i]['mean_latency_time'])

            
            if i[0] in self.patient_data and 'mean_hold_time' in self.tappy_data[i]: 

                if self.patient_data[i[0]]['Parkinsons:'] == 'True':
                    y3.append(1)
                else:
                    y3.append(0)

                x3.append(self.tappy_data[i]['mean_hold_time'])

        self.perform_singular_linear_regression([x1,y1])
        self.perform_singular_linear_regression([x2,y2])
        self.perform_singular_linear_regression([x3,y3])
    
    def perform_singular_linear_regression(self, input_output):
        
        x = numpy.array(input_output[0]).reshape((-1,1))
        y = numpy.array(input_output[1])

        model = LinearRegression().fit(x,y)
        print(model.score(x,y))

        return model


    def perform_multiple_linear_regression(self, input_output):
        # Perform multilple linearn regressions
        # Given a 2D arrat x and an output y (using 30% traingin data)
        x = numpy.array(input_output[0])
        y = numpy.array(input_output[1])

        model = LinearRegression().fit(x,y)
        print(model.score(x,y))

        return model
       
    def find_coefficient_of_determination(self, model, x, y):
        # Find the mean squared error of a model given model attributes
        # assume 30% testing data
        print(model.score(x,y))
        return model.score(x,y)

    
    def create_multiple_regressions(self):
        # Create all possible multiple regression models
        # Used mean squared error to find the most effective

        x1_x2 = []
        x2_x3 = []
        x1_x3 = []
        x1_x2_x3 = []
        

        y1_y2 = []
        y2_y3 = []
        y1_y3 = []
        y1_y2_y3 = []


        for i in self.tappy_data:

            if i[0] in self.patient_data and 'mean_flight_time' in self.tappy_data[i] and 'mean_latency_time' in self.tappy_data[i]: 
                if self.patient_data[i[0]]['Parkinsons:'] == 'True':
                    y1_y2.append(1)
                else:
                    y1_y2.append(0)
                
                x1_x2.append([self.tappy_data[i]['mean_flight_time'],self.tappy_data[i]['mean_latency_time']])
            
            

            if i[0] in self.patient_data and 'mean_flight_time' in self.tappy_data[i] and 'mean_hold_time' in self.tappy_data[i]: 

                if self.patient_data[i[0]]['Parkinsons:'] == 'True':
                    y2_y3.append(1)
                else:
                    y2_y3.append(0)

                x2_x3.append([self.tappy_data[i]['mean_flight_time'],self.tappy_data[i]['mean_hold_time']])

        
            if i[0] in self.patient_data and 'mean_latency_time' in self.tappy_data[i] and 'mean_hold_time' in self.tappy_data[i]: 

                if self.patient_data[i[0]]['Parkinsons:'] == 'True':
                    y1_y3.append(1)
                else:
                    y1_y3.append(0)

                x1_x3.append([self.tappy_data[i]['mean_latency_time'],self.tappy_data[i]['mean_hold_time']])

            
            if i[0] in self.patient_data and 'mean_latency_time' in self.tappy_data[i] and 'mean_hold_time' in self.tappy_data[i] and 'mean_flight_time' in self.tappy_data[i]: 

                if self.patient_data[i[0]]['Parkinsons:'] == 'True':
                    y1_y2_y3.append(1)
                else:
                    y1_y2_y3.append(0)

                x1_x2_x3.append([self.tappy_data[i]['mean_latency_time'],self.tappy_data[i]['mean_hold_time'], self.tappy_data[i]['mean_flight_time']])


            
        self.perform_multiple_linear_regression([x1_x2,y1_y2])
        self.perform_multiple_linear_regression([x2_x3,y2_y3])
        self.perform_multiple_linear_regression([x1_x3,y1_y3])
        self.perform_multiple_linear_regression([x1_x2_x3,y1_y2_y3])

       
        

class CreateKerasClassifier():

    def __init__(self):
        self.processed_data = PreprocessedData()
        self.patient_data = processed_data.get_patient_data()
        self.tappy_data = processed_data.get_tappy_data()


    def model_with_deep_learning(self, X, Y):

        #make a seed to be used for the model 
        seed = 6
        numpy.random.seed(seed)

        #Model is defined to active when called by Keras
        def create_model(neuron1,neuron2):
            # create model
            keras_model = Sequential()

            keras_model.add(Dense(8, input_dim = 8, kernel_initializer='normal', activation='tanh'))
            keras_model.add(Dense(4, input_dim = 8, kernel_initializer='normal', activation='tanh'))
            keras_model.add(Dense(1, activation='sigmoid'))
            
            #Configure the learning rate of the model
            adam = Adam(lr = 0.002)

            model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
            
            return model

        #Create the model 
        '''
        TODO: try different models and epoch and batch sizes (may take some time to train)
        '''
        model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 20, verbose = 0)

        #define different neurons to be used for gird search 
        neuron1= [4,8,16]
        neuron2 = [2,4,8]

        param_grid = dict(neuron1= neuron1, neuron2=neuron2)

        grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed),refit=True, verbose = 10)
        grid_results = grid.fit(X, Y)

        # return the best model
        return [grid_results.best_score_, grid_results.best_params_]


    # Get statistical results of best model 
    def get_model_data(self, grid_results):

        means = grid_results.cv_results_['mean_test_score']
        stds = grid_results.cv_results_['std_test_score']
        params = grid_results.cv_results_['params']

        results = []
        for mean, stdev, param in zip(means, stds, params):
            results.append(mean, stdev, param)
        
        return results




models = CreateLinearRegressionModels().create_multiple_regressions()





