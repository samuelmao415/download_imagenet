import os
from math import floor #floor makes a number into integer
import shutil
#https://stackoverflow.com/questions/42471570/how-to-split-documents-into-training-set-and-test-set

datadir = 'C:/Users/samuelmao/Desktop/wardrobe_detection/phone'
format = '.jpg'
split = 0.8
train_folder_path = 'C:/Users/samuelmao/Desktop/wardrobe_detection/phone/train'
test_folder_path = 'C:/Users/samuelmao/Desktop/wardrobe_detection/phone/test'

def get_training_and_testing_sets(datadir,split,train_folder_path,test_folder_path):
    def get_file_list_from_dir(datadir,format):
        all_files = os.listdir(os.path.realpath(datadir))
        data_files = list(filter(lambda file: file.endswith(format), all_files))
        return data_files

    file_list = get_file_list_from_dir(datadir,format)
    split = split
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    training_files = list(str(datadir) + '/' + x for x in training)
    testing_files = list(str(datadir) + '/' + x for x in testing)

    if not os.path.exists(train_folder_path):
        os.makedirs(train_folder_path)
        print('made folder' + train_folder_path)

    for f in training_files:
        shutil.move(f,train_folder_path)
    print('moved into training')

    if not os.path.exists(test_folder_path):
        os.makedirs(test_folder_path)
        print('made folder' + test_folder_path)

    for f in testing_files:
        shutil.move(f,test_folder_path)
    print('moved into testing')

    print ('-'*20 +'\n' + 'ta-da, magic!')

get_training_and_testing_sets(datadir,split,train_folder_path,test_folder_path)
