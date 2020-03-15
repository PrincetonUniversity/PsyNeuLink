#!/usr/bin/env python

#-------------------------------------------------------------------
# build_dict.py: format STROOP task experiment data for LVOC mechanism
# Author: Viktoria Zlatinova
#
# % python builddict.py LVOC_XOR_stimuli.csv 8
#-------------------------------------------------------------------

from sys import argv
from pandas import DataFrame, read_csv

#--------------------------------------------

# returns array representation of 1HOT vector
def to1HOT(subfeatureNum, colValue):
	i = 1
	thisarr = []
	while i < colValue:
		thisarr.append(0)
		i += 1
	thisarr.append(1)
	while (subfeatureNum - i) > 0:
		thisarr.append(0)
		i += 1
	return thisarr


# returns an array with 2 elements;
# a 1 in the first position means color naming task
def getReward(CN_rewarded):
	if CN_rewarded == 1:
		return [0,1]
	return [1,0]


# open a file corresponding to subj_id
def createFile(subj_id):
	newFile = "XOR_subj{}.txt".format(subj_id)
	return newFile


# print the input arrays to a file
def writeToFile(filename, color_stim, word_stim,
 color_task, word_task, reward):
	filename.write("{}\n".format(color_stim))
	filename.write("{}\n".format(word_stim))
	filename.write("{}\n".format(color_task))
	filename.write("{}\n".format(word_task))
	filename.write("{}\n".format(reward))


def emptyStructs(color_stim, word_stim, color_task,
 word_task, reward):
	color_stim = []
	word_stim = []
	color_task = []
	word_task = []
	reward = []
	return color_stim, word_stim, color_task, word_task, reward


# TODO add extra col for trial # & only use last 200
df = read_csv("LVOC_XOR_stimuli.csv", usecols=[0,1,3,4,5,6])

# initialize fields
color_stim = []
word_stim = []
color_task = []
word_task = []
reward = []
freq = []
trial_type = []
xor_dict = []

# number of feature stimuli
subfeature_num = 8

# start file for subject 1
subj_id = 1

for index, row in df.iterrows():
	if row[0] != subj_id:

		xor_dict.append([color_stim, word_stim, color_task,
			word_task, reward, freq, trial_type])

		color_stim = []
		word_stim = []
		color_task = []
		word_task = []
		reward = []
		freq = []
		trial_type = []

		subj_id = row[0]

	# if row[2] != 0:
	color_stim.append(to1HOT(subfeature_num, row[3]))
	word_stim.append(to1HOT(subfeature_num, row[4]))
	color_task.append([-1.32]) # given 1.32
	word_task.append([3.22]) # given 3.22
	reward.append(getReward(row[5]))
	freq.append(row[1])
	trial_type.append(row[2])

xor_dict.append([color_stim, word_stim, color_task,
			word_task, reward, freq, trial_type])

# print(xor_dict)

