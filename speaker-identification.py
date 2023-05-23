# -*- coding: utf-8 -*-
########## Extra Credit ##########################
import socket
import sys
import json
import threading
import numpy as np
import pickle
from features import FeatureExtractor
import os
import matplotlib.pyplot as plt

# Load the classifier:
output_dir = 'training_output'
classifier_filename = 'classifier.pickle'

with open(os.path.join(output_dir, classifier_filename), 'rb') as f:
    classifier = pickle.load(f)
    
if classifier == None:
    print("Classifier is null; make sure you have trained it!")
    sys.exit()
    
feature_extractor = FeatureExtractor(debug=False)
    


# ## Write the code for test code"
#Using random forest classifier - had 91% accuracy, 88% precision, 87% recall vs decision tree 86% accuracy, 74% precision, 85% recall
classnames = ["austin","runyu","silent"]
#classnames = ["austin","runyu", "ryan", "silent"]
tree = classifier

#Preprocessing
data_dir = './testdata'  # directory where the data files are stored
class_names = []  # the set of classes, i.e. speakers
data = np.zeros((0, 8002))  # 8002 = 1 (timestamp) + 8000 (for 8kHz audio data) + 1 (label)
for filename in os.listdir(data_dir):
    if filename.endswith(".csv") and filename.startswith("speaker-data"):
        filename_components = filename.split("-")  # split by the '-' character
        speaker = filename_components[2]
        print("Loading data for {}.".format(speaker))
        if speaker not in class_names:
            class_names.append(speaker)
        speaker_label = class_names.index(speaker)
        sys.stdout.flush()
        data_file = os.path.join(data_dir, filename)
        data_for_current_speaker = np.genfromtxt(data_file, delimiter=',')
        print("Loaded {} raw labelled audio data samples.".format(len(data_for_current_speaker)))
        sys.stdout.flush()
        data_for_current_speaker[:, -1] = speaker_label
        #print("speaker_label = ", speaker_label)
        data = np.append(data, data_for_current_speaker, axis=0)

#Feature extraction
n_features = 985
# default value
# n_features = 1077
print("Extracting features and labels for {} audio windows...".format(data.shape[0]))
sys.stdout.flush()
X = np.zeros((0, n_features))
y = np.zeros(0, )
feature_names = []
timestamps = []
Vals = []
ActivityTimestamps = {"austin":[],"runyu":[],"ryan":[], "silent":[]}

for i, window_with_timestamp_and_label in enumerate(data):
    window = window_with_timestamp_and_label[1:-1]
    label = window_with_timestamp_and_label[-1]
    x = feature_extractor.extract_features(window)
    prediction = tree.predict([x])
    #print(classnames[int(prediction)])
    timestamps.append(float(window_with_timestamp_and_label[0])) #Seconds
    Vals.append(window_with_timestamp_and_label[1]) 
    ActivityTimestamps[classnames[int(prediction)]].append(float(window_with_timestamp_and_label[0])) 
    #print(prediction)
    #print(class_names[int(prediction)])

plt.figure(figsize=(20,10))
plt.bar(ActivityTimestamps["austin"], np.max(Vals), color = 'cyan', label = 'austin')
plt.bar(ActivityTimestamps["runyu"], np.max(Vals), color = 'maroon', label = 'runyu')
plt.bar(ActivityTimestamps["ryan"], np.max(Vals), color = 'yellow', label = 'ryan')
plt.bar(ActivityTimestamps["silent"], np.max(Vals), color = 'orange', label = 'silent')
plt.plot(timestamps, Vals, 'r',label='Sound levels')
plt.title("Mixed Speaker Sounds")
plt.legend(loc = 'upper left')
plt.grid()
plt.show()