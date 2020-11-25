import os


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
            all_patient_tappy_data[tuple([i[0:9],i[11:14]])] = self.clean_tappy_data(i)
        
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






