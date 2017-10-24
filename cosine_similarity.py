#PHI LAM
#COEN 169
#cosine_similarity.py

#****IMPORTS****
import numpy as np
import math
import sys
from operator import itemgetter


#****VARIABLE DECLARATION****
training_rows = 200         #200 training users (indexed by <userid - 1>
training_cols = 1000        #1000 movies (indexed by <movieid - 1>)
test_rows = 100             #100 test users
test_cols = 1000            #1000 movies
training_data = [[0 for x in range(training_cols)] for y in range(training_rows)]
test_data = [[None for x in range(test_cols)] for y in range(test_rows)]

#****FUNCTION DEFINITIONS****

##############################################################################
#FUNCTION: populate()
#   Input: The training data file (train.txt)
#       Parses and converts string input to a list of lists of integers. Fills
#       indices 0-199 in training_data.
#
def populate(in_file):
    for userid, line in enumerate(in_file):     #each row is a user's ratings
        ratings = line.split()
        for movieid, rating in enumerate(ratings): #each column is a different movie
            training_data[userid][movieid] = int(rating) #user i's rating for movie j

###############################################################################
#FUNCTION: get_user_info()
#   Input: The test data file (test5.txt, test10.txt, or test20.txt)
#       test_data is a 100x1000 list of lists (100 users, 1000 ratings each)
#       with all values initialized to None (since we want 0 to signify a
#       "target index" for a predicted rating). This function parses the test
#       test file line by line, and inserts the movie rating into the matching
#       index for the given user.
#   Note: test_data is horizontally parallel to training_data (both have 1000
#       movies)
#
def get_user_info(in_file):
    global first_user
    first_user = 0
#   Actual file parsing happens here    #
    for i, line in enumerate(in_file):
        data = line.split()
        if first_user == 0:
            first_user = int(data[0])
        user = int(data[0]) - first_user  #userid is first_user-indexed, list is 0-indexed
        movie = int(data[1]) - 1      #movieid is 1-indexed, list is 0-indexed
        rating = int(data[2])
        test_data[user][movie] = rating

##############################################################################
#FUNCTION: dot()
#   Input: two lists
#   Custom dot product: skips dimension if either of the vectors
#       is missing that dimension, I.E. if "None" or 0 is found.
#
def dot(list1, list2):
    assert len(list1) == len(list2), "Dot product: list lengths must be the same!"
    result = 0
    for i, dim in enumerate(list1):
        if list1[i] == 0 or list1[i] == None:
            continue
        if list2[i] == 0 or list2[i] == None:
            continue
        else:
            result += list1[i]*list2[i]
    return result

############################################################################
#FUNCTION: cos_sim_denom()
#   Input: two lists
#       Compute the vector lengths of the two lists, and multiply them together.
#       Like the custom dot product above, skip dimensions that aren't shared
#       by both vectors.
#
def cos_sim_denom(list1, list2):
    assert len(list1) == len(list2), "Dot product: list lengths must be the same!"
    sum1 = 0
    sum2 = 0
    result = 0
    for i, dim in enumerate(list1):
        if list1[i] == 0 or list1[i] == None:
            continue
        if list2[i] == 0 or list2[i] == None:
            continue
        else:
            sum1 += list1[i]**2
            sum2 += list2[i]**2
    result = math.sqrt(sum1)*math.sqrt(sum2)
    return result

############################################################################
#FUNCTION: cosine_similarity
#   Input: active user (one row from test_data)
#       Calculates the vector cosine similarity of the active user
#       against every user in training_data, then outputs a list of
#       predicted ratings for each 0 in active_user
#
def cosine_similarity(active_user):
    weight_list = []        #List of cosine similarities compared
                                #   to active user. Each index corresponds
                                #   to a user, each value is that user's
                                #   similarity to the active user.

    ranked_neighbors = []       #List of pairs: [cos_sim, userid]. Created
                                #   from weight_list
    predicted_ratings = [0]*test_cols

###########################################################################
#   Part 1: Cosine Vector Similarity
#       Numerator: inner product of active_user and current user in training
#           data.
#       Denominator: product of square roots of the sums of the squares of a vector's
#           components (i.e. product of vector lengths)
#       Fills weight_list, which is parallel with training_data
#
    for user in training_data:  #each user is a list of all that user's ratings
        num = dot(active_user, user)          #numerator is dot product
        denom = cos_sim_denom(active_user, user)
        if denom == 0:
            weight_list.append(0)
#            print("cos_sim=0")
            continue
        cos_sim = num/denom
        weight_list.append(cos_sim)
        # print(cos_sim)
        # print("num:{} denom:{} cos_sim:{}".format(num, denom,cos_sim))

#############################################################################
#   Part 1.5: Sort weight_list and take top-K neighbors
#        Let k = 150
#       1) Get index of largest pearson correlation weight
#       2) Save the largest weight value
#       3) Clear the weight but do not remove (preserve indices)
#       3) Add [weight, index] to ranked_neighbors
#       4) Go to step 1, repeat for k iterations
#   This step destroys the weight_list as it existed (parallel with
#       relevant_users), but we no longer need it once we have the
#       top-K neighbors so long as we preserve the indices so that
#       we can refer back to relevant_users
#   ranked_neighbors is not parallel with relevant_users, but
#       storing the index in the second part of the pair allows us to refer
#       to the ratings in relevant_users easily
#
    while len(ranked_neighbors) < 160:
       userid = weight_list.index(max(weight_list))
       weight = weight_list[userid]
       weight_list[userid] = 0
       ranked_neighbors.append([weight, userid])

    # for userid, weight in enumerate(weight_list):
    #    if weight > 0.1 and weight < 1.0:
    #        ranked_neighbors.append([weight, userid])
    #    else:
    #        ranked_neighbors.append([0, userid])
    # sorted(ranked_neighbors, key = itemgetter(0))
#   print(ranked_neighbors)

##############################################################################
#   Part 2: Rating Prediction
#   Compute weighted average of similar users' ratings for each 0 in
#       the active user list
#       1) Iterate by movieid in active_user
#       2) If the rating is a 0, we need to provide a predicted rating.
#       3) Refer to ranked_neighbors to obtain the cos_sim weights and the
#           indices of the rating belonging to the current relevant neighbor
#       4)
#       Notes:
#       active_user[movieid] = rating
#       relevant_user[pair[1]][movieid] = rating for current ranked_neighbor

    for movieid, a_rating in enumerate(active_user):
        if a_rating == 0:             #if active user has not rated this movie
            sum1 = 0
            sum2 = 0
            result = 0
            # for userid, cos_sim in enumerate(weight_list):
            for pair in ranked_neighbors:
                t_rating = training_data[pair[1]][movieid]
                cos_sim = pair[0]
                if t_rating != 0:   #ensure training user has rated the target movie.
                    sum1 += cos_sim * t_rating #cos_sim*neighbor rating
                    sum2 += cos_sim
                    # print(cos_sim)
            if sum2 == 0:           #avoid division by 0
                result = 0
            else:
                result = round(sum1/sum2)

            if result == 0:     #Edge case: predicted rating is 0
                result = 3
            # print("sum1:{} | sum2:{}".format(sum1, sum2))
#            print(sum1/sum2)
            predicted_ratings[movieid] = result
    return(predicted_ratings)

##############################################################################
# "main"
if len(sys.argv) < 4:
    print("Usage: cosine_similarity.py training_data test_data out_file\n")
    exit()
try:
    f1 = open(sys.argv[1])
    f2 = open(sys.argv[2])
    f3 = open(sys.argv[3], "w")
except:
    print("Usage: arguments must be text files")
    exit()

print("Processing training data...")
populate(f1)
print("Processing test data...")
get_user_info(f2)
# ######### TEST CODE ############
# john = [9,3,0,0,5]
# mary = [10,3,8,0,5]
# print("john*mary = {}".format(dot(john,mary)))
# cos_sim = dot(john, mary)/(cos_sim_denom(john, mary))
# print(cos_sim)
# exit()
print("Predicting ratings...")
for userid, user in enumerate(test_data):
    predicted_ratings = cosine_similarity(user)
    for movieid, rating in enumerate(predicted_ratings):
        if rating == 0:
            continue
        else:
            f3.write("{} {} {}\n".format(userid+first_user, movieid+1, rating))
f1.close()
f2.close()
f3.close()
