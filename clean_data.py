import os

all_patients_demographic_data = {}
all_patient_tappy_data = {}


def create_demographic_csv(file_name):
    first= open('./Archived users/' + file_name, 'r')
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


def create_tappy_data_csv(file_name):
    second= open('./Tappy Data/' + file_name)
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




archived_users = os.listdir('./Archived users')
tappy_data = os.listdir('./Tappy Data')

for i in archived_users:
    all_patients_demographic_data[i[5:15]] = create_demographic_csv(i)



for i in tappy_data:
    all_patient_tappy_data[tuple([i[0:9],i[11:14]])] = create_tappy_data_csv(i)

print(len(all_patient_tappy_data))








