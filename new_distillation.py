import gensim
#import sklearn
import sys
import os
import pickle
from scipy import spatial

if len(sys.argv) > 1:
    model_num = sys.argv[1]
else:
    print ("Using : python new_distillation.py [model_number]")
    sys.exit()

def cos_sim(list_1, list_2):
    return 1 - spatial.distance.cosine(list_1, list_2)

# Generate output list
output_file = open("distill_files/output.txt{}".format(str(model_num)), 'r')

output_list = []
all_list = []

for line in output_file:
    all_list.append(line)
    if line not in output_list:
        output_list.append(line)
output_file.close()
print ("output list length :", len(output_list))

# Find index in dictionary
index_dictionary = {}

index = 0
for line in all_list:
    if line not in index_dictionary:
        index_dictionary[line] = [index]
    else:
        index_dictionary[line].append(index)
    index += 1
index_dictionary = sorted(index_dictionary.items(), key=lambda x: len(x[1]),reverse=True)
print ("index list length :", len(index_dictionary))

# Calculating input similarity
similarity_list = []
input_file = open("distill_files/input_emb_{}.pkl".format(model_num), 'rb')
#input_file = open("enc_embedding.pkl", 'rb')
input_embed = pickle.load(input_file)
input_file.close()

sim_file = open("distill_files/similarity_list_{}".format(model_num), 'w')

i = 0
threshold_num = 20

for line in index_dictionary:
    length = len(line[1])
    if length<=threshold_num:break
    ave_sim = 0
    count = 0
    for r_index in range(len(line[1])):
        if r_index != len(line[1]):
            for c_index in range(r_index, len(line[1])):
                sim = cos_sim(input_embed[line[1][r_index]], input_embed[line[1][c_index]])
                ave_sim += sim
                count += 1
    ave_sim = float((ave_sim-length)/(count-length))
    print ("ave_sim : ", ave_sim)
    print ("count : ", count-length)
    print ("length : ", length)
    similarity_list.append(ave_sim)
    sim_file.write(str(ave_sim))
    sim_file.write("\t")
    sim_file.write(str(line[1]))
    sim_file.write("\t")
    sim_file.write(str(line[0]).strip())
    sim_file.write("\n")

sim_file.close()

