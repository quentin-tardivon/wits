# What is this Sound?
## Maynooth University - CS401 Project
![image](./maynooth-logo.png)

### Quentin Tardivon - 17182786
### Niko Karuhla - 

## 1. Project Goals

The goal of this project was to discover the usage of Tensorflow and to work with data with the Tfrecord format. 
We used for that the Audioset project, gathering 10 seconds audio clip extracted from Youtube with labels for each one.
The final goal was to try to create a classifier able to tell apart audio files containing speech from others sounds.

## 2. Realized Work

  ### 1. Extracting data

  The main goal was to understand the format of the data and how to work with it. The main advantage of tfrecord format is that you can 
  save large amount of data with smaller disk usage. Indeed, the complete audioset balanced files is around 14mb against several gigabits for
  the wav audio files. 

  Google use for this type of file the protocol buffer which allow to organize the data for easier parsing and exchange.
  The main problem for this step was that the data format was not clearly described on the project page and we had to guess the data structure 
  to parse the data. Indeed, if you do not have information about how the data was organized, it is impossible to retrieve your data from 
  a tfrecord file.
  ### 2. Creating new data
  In order to test the model, we wanted to create new data with the same data format as the Audioset project but with our own wav files. 
  For that we use the vggish model available on github and tweak it in order to work with our wav files.
  ### 3. Learning phase
  We encounter a number of problem during this phase linked to Tensorflow and its documentation. 
  
  In the end, we used Tflearn, a high-level wrapper for Tensorflow which simplify the creation of classifiers and computational graphs. 
  For the training, we use a simple convolutional network fully connected for the first tests. We try different batch sizes and epoch with Adam
  optimizer. We used Tensorboard to vizualize learning data. 
  We were able to observe that the learning was non conclusive. Indeed, we can observe that the accuracy jump to 100% very fast.

## 3. Possible Improvements

## 4. Conclusion
