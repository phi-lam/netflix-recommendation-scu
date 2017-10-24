#PHI LAM
#COEN 169
#item_pearson.py

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
#Initialize the transposed lists
inv_training_data = [[0 for x in range(training_rows)] for y in range(training_cols)]
inv_test_data = [[None for x in range(test_rows)] for y in range(test_cols)]

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
            inv_training_data[movieid][userid] = int(rating)

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
def adj_dot(list1, list2):
    assert len(list1) == len(list2), "Dot product: list lengths must be the same!"
    result = 0
    if list1 == list2:
        return 0
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
#FUNCTION: item_pearson
#   Input: active user (one row from test_data)
#       Calculates the vector cosine similarity of the active user
#       against every user in training_data, then outputs a list of
#       predicted ratings for each 0 in active_user
#
def item_pearson(active_user):
    target_movies = []          #1-D list stores indices of targets for prediction
    active_movies = []          #1-D list stores indices of active_user's known ratings

    print("new active user")
    for movieid, rating in enumerate(active_user):
        if rating == None:
            continue
        elif rating == 0:
            target_movies.append(movieid)
        else:
            inv_training_data[movieid].append(rating)
            active_movies.append(movieid)
            print("known movie: ", movieid, "rating: ", rating)

    weight_list = [[]]

    ranked_neighbors = []       #List of pairs: [cos_sim, userid]. Created
                                #   from weight_list
    predicted_ratings = [0]*test_cols

###########################################################################
#   Part 1: Item-Based Pearson Similarity
#     1) Select movie from active_movies (in order)
#     2) Compare that active movie against each movie in the transposed
#           training data
#     3) Calculate the adjusted cosine sim for those two movies by:
#           a) Starting from the first user in the training data
#           b) Subtract the user's average rating from its ratings for
#               the active movie and the training movie. Plug into the
#               similarity formula.
#           c) Move to the next user, continue until all training users used.
#           d) Summing the results as per the formula gives the similarity
#               between the active movie and the current training movie.
#     4) Repeat for each training movie, and then repeat everything for
#           each active movie.
#     5) weight_list is a 2-D list, where i is the sim_list for each active movie.
#           j is the similarity between the active movie and the jth movie

    for i, movieid in enumerate(target_movies):           #Each member takes a turn as active movie
        weight_list.append([])
        for known_movie in active_movies:   #Active movie is compared against each training movie
            num = 0
            denom_a = 0
            denom_t = 0
            for userid, user in enumerate(training_data):      #The formula moves user by user
                movie_avg_i = training_movie_avgs[movieid]
                movie_avg_j = training_movie_avgs[known_movie]
                a_movie = inv_training_data[movieid]
                t_movie = inv_training_data[known_movie]

                var_a = a_movie[userid] - movie_avg_i
                var_t = t_movie[userid] - movie_avg_j
                num += (var_a) * (var_t)
                denom_a += var_a ** 2
                denom_t += var_t ** 2
            denom = math.sqrt(denom_a) * math.sqrt(denom_t)
            if denom == 0:
                weight = 0
            else:
                weight = num/denom
            weight_list[i].append(weight)

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
#
    for i, target_movie in enumerate(target_movies):
        sum1 = 0
        sum2 = 0
        result = 0
        for j, known_movie in enumerate(active_movies):
            sum1 += weight_list[i][j] * active_user[known_movie]
            sum2 += weight_list[i][j]
        if sum2 == 0:
            result = 0
        else:
            result = round(sum1/sum2)
        if result == 0:
            result = 3
        predicted_ratings[target_movie] = result
    return(predicted_ratings)

##############################################################################
# "main"
if len(sys.argv) < 4:
    print("Usage: item_pearson.py training_data test_data out_file\n")
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

print("Predicting ratings...")

training_movie_avgs = []
for movieid, movie in enumerate(inv_training_data):      #The formula moves user by user
    movieavg = pearson_avg(movie)
    training_movie_avgs.append(movieavg)

for userid, user in enumerate(test_data):
    predicted_ratings = item_pearson(user)
    for movieid, rating in enumerate(predicted_ratings):
        if rating == 0:
            continue
        else:
            f3.write("{} {} {}\n".format(userid+first_user, movieid+1, rating))
f1.close()
f2.close()
f3.close()
