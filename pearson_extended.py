#PHI LAM
#COEN 169
#pearson_extended.py

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
global first_user              #Used to shift indices for output

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
#   Description:
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
    for line in in_file:
        data = line.split()
        if first_user == 0:
            first_user = int(data[0])
        user = int(data[0]) - first_user  #userid is first_user-indexed, list is 0-indexed
        movie = int(data[1]) - 1      #movieid is 1-indexed, list is 0-indexed
        rating = int(data[2])
        test_data[user][movie] = rating

###############################################################################
#FUNCTION: calculate_IUF()
#   Input: two lists, active user and training data
#   Description:
#       For each column, count the number of users that have a nonzero rating.
#       Calculate the IUF for that column, add it to the IUF list, and repeat
#       for the next column.
#   Returns an iuf_list, a 1x1000 array of IUF values for each movie column.
#   This IUF list is relevant as long as the same active_user is being processed.
#
def calculate_IUF(active_user, training_data):
    print("Calculating IUF values...")
    iuf_list = []
    for movieid, a_rating in enumerate(active_user):
#1) Count the active_user for the total user count, and increment rating_count
#           if it has a nonzero rating.
        rating_count = 0
        user_count = 1
        if a_rating != 0 or a_rating != None:
            rating_count += 1
#2) For the same movieid (column), check each user in the training data
#           for nonzero ratings.
        for user in training_data:
            user_count += 1
            t_rating = user[movieid]
            if t_rating != 0 or t_rating != None:
                rating_count += 1
#3) Avoid division by 0
        if rating_count == 0:
            iuf_list.append(0)
            continue
#4) Calculate IUF and add to IUF list.
        iuf = math.log(user_count/rating_count)
        iuf_list.append(iuf)
    return iuf_list

##############################################################################
#FUNCTION: dot() [helper]
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
#FUNCTION: pearson_avg() [helper]
#   Input: one list
#       Computes the average rating of a user for the purpose of computing
#       the Pearson Correlation. Skips over 0s and Nones when calculating the
#       length of the user.

def pearson_avg(list1):
    sum1 = 0
    count = 0
    result = 0
    for i in range(len(list1)):
        if list1[i] == 0 or list1[i] == None:
            continue
        else:
            sum1 += list1[i]
            count += 1
    if count == 0:
        print("Calculated pearson_avg is 0")
        return 0
    result = sum1/count
    return result

############################################################################
#FUNCTION: pearson_correlation
#   Input: active user (one row from test_data)
#       Calculates the vector cosine similarity of the active user
#       against every user in training_data, then outputs a list of
#       predicted ratings for each 0 in active_user
#
def pearson_correlation(active_user):
    weight_list = []        #List of cosine similarities compared
                                #   to active user. Each index corresponds
                                #   to a user, each value is that user's
                                #   similarity to the active user.

    ranked_neighbors = []       #List of pairs: [cos_sim, userid]. Created
                                #   from weight_list
    active_avg = pearson_avg(active_user)
    #print(active_avg)

    predicted_ratings = [0]*test_cols

    iuf_list = calculate_IUF(active_user, training_data)

###########################################################################
#   Part 1: Pearson Correlation Weight Calculation
#       Numerator: sum((active_user's rating - avg) * (test_user's rating-avg))
#       Denominator: product of active and test users':
#    sqrt(sum(active user's rating - avg)) * sqrt(sum(test user's rating - avg))
#       Fills weight_list, which is parallel with relevant_users
#
#    print("Pearson calculation")

    for userid, user in enumerate(training_data):
        num = 0
        denom_a = 0
        denom_t = 0
        train_avg = training_user_avgs[userid]
        # print(train_avg)
        #avgs = pearson_avg(active_user, user)
        for movieid, rating in enumerate(user):     #skip non-shared dimensions
            if active_user[movieid] == None or active_user[movieid] == 0:
                continue
            if rating == 0 or rating == None:
                continue
            var_a = (active_user[movieid]*iuf_list[movieid] - active_avg) #value for active user
            var_t = (rating * iuf_list[movieid] - train_avg)               #value for test_user
            num += var_a * var_t
            denom_a += var_a**2
            denom_t += var_t**2
            # print("active rating: ", active_user[movieid])
            # print("active avg: ", active_avg)
            # print("rating: ", rating)
            # print("train_avg: ", train_avg)
        denom = math.sqrt(denom_a) * math.sqrt(denom_t)
        if denom == 0:
            weight_list.append(0)
            continue
        weight = num/denom
        #Case Amplification here
        weight = weight * (math.fabs(weight) ** 1.5)
        weight_list.append(weight)
        print("num:{} denom:{} pearson:{}".format(num, denom,weight))
#        print("pearson correlation: {}".format(weight))

#############################################################################
#   Part 1.5: Sort weight_list and take top-K neighbors
#        Take top users with weights greater than 0.5

    for userid, weight in enumerate(weight_list):
        if math.fabs(weight) > 0.5:
            ranked_neighbors.append([weight, userid])
        else:
            ranked_neighbors.append([0, userid])

    sorted(ranked_neighbors, key = itemgetter(0))

##############################################################################
#   Part 2: Rating Prediction
#   Compute weighted average of similar users' ratings for each 0 in
#       the active user list
#       1) Iterate by movieid in active_user
#       2) If the rating is a 0, we need to provide a predicted rating.
#       3) Refer to ranked_neighbors to obtain the pearson weights and the
#           indices of the ratings belonging to the current relevant neighbor
#       4) Both IUF and Case Amplification happen here, when the training rating
#           is pulled from the training data.
#       Notes:
#       active_user[movieid] = rating
#       relevant_user[pair[1]][movieid] = rating for current ranked_neighbor
#
#    print("Rating prediction")
    for movieid, a_rating in enumerate(active_user):
        if a_rating == 0:
            result = 0
            num = 0
            denom_a = 0
            denom_t = 0
            for pair in ranked_neighbors:
                #IUF here
                train_rating = training_data[pair[1]][movieid]
                train_avg = pearson_avg(training_data[pair[1]])
                pearson = pair[0]

                pearson = pearson
                #Actual calculations
                if train_rating != 0:
                    num += pearson * (train_rating - train_avg)
                    denom += math.fabs(pearson)
            if denom == 0:           #avoid division by 0
                result = 0
            else:
                result = round(active_avg + (num/denom))
            if result == 0:     #Edge case
                result = 3
            predicted_ratings[movieid] = result
    return(predicted_ratings)
##############################################################################
# "main"
if len(sys.argv) < 4:
    print("Usage: pearson_correlation.py training_data test_data out_file\n")
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

training_user_avgs = []
for userid, user in enumerate(training_data):
    useravg = pearson_avg(user)
    training_user_avgs.append(useravg)

print("Predicting ratings...")
for userid, user in enumerate(test_data):
    predicted_ratings = pearson_correlation(user)
    for movieid, rating in enumerate(predicted_ratings):
        if rating == 0:
            continue
        else:
            f3.write("{} {} {}\n".format(userid+first_user, movieid+1, rating))
f1.close()
f2.close()
f3.close()
