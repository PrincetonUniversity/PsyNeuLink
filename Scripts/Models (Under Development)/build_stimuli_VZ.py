# color naming: 1, word reading: -1

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

# initialize fields
color_stim = []
word_stim = []
color_task = []
word_task = []
reward = []
xor_dict = []

# number of feature stimuli
total_features = 8


subj_id = 1
trial_num = 0
feature_num = 1

while (trial_num < 100):
# while (trial_num <= 65):
# 	if trial_num == 65:
# 		trial_num = 0
# 		if feature_num == 8:
# 			feature_num = 1
# 			xor_dict.append([color_stim, word_stim, color_task, word_task, reward])
# 			if subj_id == 30:
# 				break
# 			else:
# 				subj_id += 1
# 				color_stim = []
# 				word_stim = []
# 				color_task = []
# 				word_task = []
# 				reward = []
# 				freq = []
# 				trial_type = []
# 		else:
# 			feature_num += 1
# 		continue
	color_stim.append(to1HOT(total_features, 1)) # congruent stimuli, x65
	word_stim.append(to1HOT(total_features, 8))
	color_task.append([1.32]) # given 1.32
	word_task.append([-3.22]) # given 3.22
# reward.append([1,0])
# if feature_num <= 2:
	# reward.append([1,0]) # CN reward
# else:
# reward.append([0,5]) # WR reward
	reward.append([5,0]) # CN reward
	trial_num += 1
xor_dict.append([color_stim, word_stim, color_task, word_task, reward])

# print("new color len: ", color_stim)
# print("new word len: ", len(word_stim))
# print("new color task len: ", len(color_task))
# print("new word task len: ", len(word_task))
# print("new reward len: ", len(reward))
# print("reward: ", reward)
# print(xor_dict)