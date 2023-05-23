## Speaker Identification using Machine Learning
In this project, we aim to solve speaker identification problem using machine learning techniques. We extract informative audio features from labeled audio data, and train a classifier to predict the speaker.

## Background
Speaker identification has a wide range of applications including authentication, forensic analysis and mobile health analytics. It can help in identifying conversation patterns which can be used to infer social habits over time.

## Dependencies
Python Speech Features: Install using pip install python_speech_features
Audiolazy: Install using pip install audiolazy
Data Collection
The dataset consists of audio data from at least 3 different speakers. Each audio session is at least 3 minutes long. There's also a no-speaker session to capture different acoustic environments.

## Naming Convention
The audio files are named as speaker-data-*-#.wav, where * is replaced by the speaker’s name or identifier, and # is the index of that speaker’s data.

## Feature Extraction
The features.py script is used to extract features from the audio data. It computes Formant Features and Delta Coefficients, which are highly informative for speaker identification.
