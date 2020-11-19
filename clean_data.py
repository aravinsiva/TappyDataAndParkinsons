import os
import  csv

def create_demographic_csv(file_name):
    first= open('./archived_users/' + file_name, 'r')
    lines = first.readlines()
    print(file_name)
    for i in lines:
        value = i[i.index(':')+1:]
        key = i[:i.index(':')+1]
        dictionary = {}

        for j in range(len(key)):
            dictionary[key[j]] = value[j]

        print(key)
        print(dictionary)
        with open('demographic_file.csv', mode='w') as demo_file:
            demo_file_writer = csv.DictWriter(demo_file, key)
            demo_file_writer.writerow(dictionary)



def create_tappy_data_csv(file_name):
    second= open('./tappy_data/' + file_name)
    lines = second.readline()
    print(lines)


archived_users = os.listdir('./archived_users')
tappy_data = os.listdir('./tappy_data')

#for i in archived_users:
create_demographic_csv(archived_users[0])

# for i in tappy_data:
create_tappy_data_csv(tappy_data[0])





    # Include conde to read file and writwe to csv



