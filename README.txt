This is a supervised machine learning program that contains two main files - createModel & speedUpper. 

The purpose of speedUpper is to analyse audio files (designed for music podcasts) to extract which portions of the file contains content defined as "music" and
 "non-music". speedUpper will then speed up the "non-music" portion of the files to a desired rate and leave the "music" portion at the correct rate. It will then 
output the modified to the desired location.

createModel is a supervised machine learning program which takes an audio file and text file as input and creates a model "music_model.pkl". This model will be the
basis that speedUpper will use to modify input audio files. An example text file is given to show the format the supervised file must be, however a model is already 
provided.


 