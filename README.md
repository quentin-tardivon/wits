# What Is This Sound?

## Project Brief
Wits (What Is This Sound?) is a Machine Learning project for Maynooth University CS401.

## Introduction

The goal of this project is to use Neural Network to train a model for sound recognition in 
noisy envrironment (real-world environment). If the training is successful, we can use
this model in a smartphone app for people with hearing loss to understand the sounds around them.

## Methods

For this project, we will try to use Neural Network within TensorFlow to train a model.

## Materials

This project will used Audio Dataset create by Google for the sound recognition in youtube video.

## Expected results

A success for this project will be to recognize the differences between global category of sounds:
- noise
- voices
- animal

A more precise classification may be difficult to achieve.

## Risk assessment

The main difficulty will be to transform the data for a suitable use in neural network.
Two approachs will be try:
- Work with raw audio data
- Transform audio data in image data (in spectrogram for example)

We will try to compare the two methods.

## Acknowledgments

This project is largely inspired by the work at Google Research (https://research.google.com/audioset/).

## Statement

We will keep all the source code in git and commit and push often and arrange for you to have access to the repository.

## Authors

* [Niko Karhula](https://github.com/niko-ka)
* [Quentin Tardivon](https://github.com/quentin-tardivon)

## Resources
* [Dataset](https://research.google.com/audioset//download.html)
* [Inspiration Video](https://www.youtube.com/watch?v=F0-RiOqgG68)
* [tensorflow](https://www.tensorflow.org/)