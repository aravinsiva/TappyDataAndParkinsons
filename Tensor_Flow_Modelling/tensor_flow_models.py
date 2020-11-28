

#Configure python imports as well as Keras Classifier
import numpy
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import *
from keras.layers import *
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import *



def model_with_deep_learning(X, Y):

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
def get_model_data(grid_results):

    means = grid_results.cv_results_['mean_test_score']
    stds = grid_results.cv_results_['std_test_score']
    params = grid_results.cv_results_['params']

    results = []
    for mean, stdev, param in zip(means, stds, params):
        results.append(mean, stdev, param)
    
    return results