#!/usr/local/bin/python3
# route.py : Find routes through maps
#
# Code by: Akshat Arvind (aarvind@iu.edu), Aniket Kale (ankale@iu.edu), Rahul Shamdasani (rshamdas@iu.edu)
#
# Based on skeleton code by V. Mathur and D. Crandall, January 2021
#


# !/usr/bin/env python3
import sys
import heapq
import numpy as np
import pandas as pd
from copy import deepcopy
import time

# import regex as re


# class Cities:
#     def __init__(self, citygps, road_segments):
#         self.citygps = citygps
#         self.road_segments = road_segments
#         self.Quebec_la = 46.829853
#         self.Quebec_lo = -71.254028
#
#     def citygps_nearest_highway(self, highway):
#         length = self.road_segments[self.road_segments['first_city'] == highway][
#                      ['second_city', 'length']].values.tolist() + \
#                  self.road_segments[self.road_segments['second_city'] == highway][
#                      ['first_city', 'length']].values.tolist()
#         print(length)
#         minlen = 999
#         minlen_city = None
#         for l in length:
#             if '&' not in l[0]:
#                 if l[1] < minlen:
#                     minlen = l[1]
#                     minlen_city = l[0]
#
#         if minlen_city != None:
#             print(minlen_city)
#             return self.citygps[self.citygps['City'] == minlen_city][['la', 'lo']].values.tolist()
#         else:
#             print('No city found in citygps_nearest_highway')
#             return []
#
#     def distance_heuristic(self, succ_city, goal_city):
#         # calculate dist between curr and succ
#         try:
#             return np.sqrt(np.square(self.citygps[self.citygps['City'] == goal_city]['la'].iloc[0] - \
#                                      self.citygps[self.citygps['City'] == succ_city]['la'].iloc[0]) \
#                            + np.square(self.citygps[self.citygps['City'] == goal_city]['lo'].iloc[0] - \
#                                        self.citygps[self.citygps['City'] == succ_city]['lo'].iloc[0]))
#         except IndexError:
#
#             print('City: ' + succ_city + ' or ' + goal_city + ' NOT FOUND in distance heuristic.')
#             # print((succ_city, goal_city))
#             print('---')
#
#             if '_Quebec' in succ_city:
#                 print('Quebec in succ_city')
#                 try:
#                     return np.sqrt(np.square(self.citygps[self.citygps['City'] == goal_city]['la'].iloc[0] - self.Quebec_la) \
#                                    + np.square(self.citygps[self.citygps['City'] == goal_city]['lo'].iloc[0] - self.Quebec_lo))
#                 except IndexError:
#                     print('_Quebec but goal_city: (' + goal_city + ') NOT FOUND')
#                     return 0
#             if '_Quebec' in goal_city:
#                 print('Quebec in goal_city')
#                 try:
#                     return np.sqrt(np.square(self.citygps[self.citygps['City'] == succ_city]['la'].iloc[0] - self.Quebec_la) \
#                                    + np.square(self.citygps[self.citygps['City'] == succ_city]['lo'].iloc[0] -  self.Quebec_lo))
#                 except IndexError:
#                     print('_Quebec but succ_city: (' + succ_city + ') NOT FOUND')
#                     return 0
#
#             #  ---
#             # if '_Quebec' in succ_city:
#             #     print('_Quebec')
#             #     try:
#             #         return np.sqrt(np.square(self.citygps[self.citygps['City'] == goal_city]['la'].iloc[0] - \
#             #                              self.Quebec_la) \
#             #                    + np.square(self.citygps[self.citygps['City'] == goal_city]['lo'].iloc[0] - \
#             #                                self.Quebec_lo))
#             #     except IndexError:
#             #         print('_Quebec but goal_city: (' + goal_city + ') NOT FOUND')
#             #         return 0
#             # if '_Quebec' in goal_city:
#             #     try:
#             #         if '&' not in succ_city:
#             #             return np.sqrt(np.square(self.citygps[self.citygps['City'] == succ_city]['la'].iloc[0] - \
#             #                                self.Quebec_la) \
#             #                           + np.square(self.citygps[self.citygps['City'] == succ_city]['lo'].iloc[0] - \
#             #                                       self.Quebec_lo))
#             #         else:
#             #             nearest_lalo = self.citygps_nearest_highway(succ_city)
#             #             if nearest_lalo != []:
#             #                 return np.sqrt(np.square(nearest_lalo[0][0] - \
#             #                                      self.Quebec_la) \
#             #                            + np.square(nearest_lalo[0][1] - \
#             #                                        self.Quebec_lo))
#             #     except IndexError:
#             #         print('_Quebec but succ_city: (' + succ_city + ') NOT FOUND')
#             #         return 999
#             # if '&' in goal_city:
#             #     nearest_lalo = self.citygps_nearest_highway(succ_city)
#             #     if nearest_lalo != []:
#             #         return np.sqrt(np.square(nearest_lalo[0][0] - \
#             #                                  self.citygps[self.citygps['City'] == succ_city]['la'].iloc[0]) \
#             #                        + np.square(nearest_lalo[0][1] - \
#             #                                    self.citygps[self.citygps['City'] == succ_city]['la'].iloc[0]))
#             # if '&' in succ_city:
#             #     nearest_lalo = self.citygps_nearest_highway(succ_city)
#             #     if nearest_lalo != []:
#             #         return np.sqrt(np.square(nearest_lalo[0][0] - \
#             #                              self.citygps[self.citygps['City'] == goal_city]['la'].iloc[0]) \
#             #                    + np.square(nearest_lalo[0][1] - \
#             #                                self.citygps[self.citygps['City'] == goal_city]['la'].iloc[0] ))
#             return 0
#
#     def distance_cost(self, curr_city, succ_city):
#         # calculate dist between curr and succ
#         try:
#             return self.road_segments[((self.road_segments['first_city'] == curr_city) & \
#                                        (self.road_segments['second_city'] == succ_city)) | \
#                                       ((self.road_segments['second_city'] == curr_city) & \
#                                        (self.road_segments['first_city'] == succ_city))
#                                       ]['length'].iloc[0]
#         except IndexError:
#             print('City not found in distance cost ')
#
#             return 0
#
#     def time_heuristic(self, curr_city, succ_city, goal_city):
#         # calculate speed of  CURR and
#         # temp = self.road_segments[(self.road_segments['first_city'] == curr_city) & (self.road_segments['second_city'] == succ_city)]['speed_limit']
#         # if temp.shape[0] == 0:
#         #     temp = self.road_segments[ (self.road_segments['second_city'] == curr_city) & (self.road_segments['first_city'] == succ_city)]['speed_limit']
#         speed_limit = self.road_segments[((self.road_segments['first_city'] == curr_city) & \
#                             (self.road_segments['second_city'] == succ_city)) | \
#                            ((self.road_segments['second_city'] == curr_city) & \
#                             (self.road_segments['first_city'] == succ_city))]['speed_limit'].iloc[0]
#         try:
#
#             return (np.sqrt(np.square(self.citygps[self.citygps['City'] == goal_city]['la'].iloc[0] - \
#                                       self.citygps[self.citygps['City'] == succ_city]['la'].iloc[0]) \
#                             + np.square(self.citygps[self.citygps['City'] == goal_city]['lo'].iloc[0] - \
#                                         self.citygps[self.citygps['City'] == succ_city]['lo'].iloc[0]))) / speed_limit
#
#         except IndexError:
#             print('City: ' + succ_city + ' or ' + goal_city + ' NOT FOUND in time heuristic.')
#             # print((succ_city, goal_city))
#             print('---')
#             if '_Quebec' in succ_city:
#                 print('_Quebec')
#                 try:
#                     return np.sqrt(np.square(self.citygps[self.citygps['City'] == goal_city]['la'].iloc[0] - self.Quebec_la) \
#                                    + np.square(self.citygps[self.citygps['City'] == goal_city]['lo'].iloc[0] - self.Quebec_lo))/ speed_limit
#                 except IndexError:
#                     print('_Quebec but goal_city: (' + goal_city + ') NOT FOUND')
#                     return 0
#             if '_Quebec' in goal_city:
#                 try:
#                     return np.sqrt(np.square(self.citygps[self.citygps['City'] == succ_city]['la'].iloc[0] - self.Quebec_la) \
#                                    + np.square(self.citygps[self.citygps['City'] == succ_city]['lo'].iloc[0] -  self.Quebec_lo))/ speed_limit
#                 except IndexError:
#                     print('_Quebec but succ_city: (' + succ_city + ') NOT FOUND')
#                     return 0
#             return 0
#
#     def time_cost(self, curr_city, succ_city):
#         # calculate speed of  CURR and
#         try:
#             return (self.road_segments[((self.road_segments['first_city'] == curr_city) & \
#                             (self.road_segments['second_city'] == succ_city)) | \
#                            ((self.road_segments['second_city'] == curr_city) & \
#                             (self.road_segments['first_city'] == succ_city))]['length'].iloc[0]) / \
#                    (self.road_segments[((self.road_segments['first_city'] == curr_city) & \
#                                        (self.road_segments['second_city'] == succ_city)) | \
#                                       ((self.road_segments['second_city'] == curr_city) & \
#                                        (self.road_segments['first_city'] == succ_city))]['speed_limit'].iloc[0])
#
#         except IndexError:
#             print('City ('+curr_city + ' or ' + succ_city +') NOT FOUND in time cost ')
#             return 0
#
#     def successor_cities(self, curr_city):
#         """
#         Returns successor cities for 'curr_city' (Type: List of List of cities with length, 'speed_limit', 'highway' )
#
#         """
#         # print(road_segments[road_segments['first_city'] == curr_city])
#         # print(road_segments[road_segments['second_city'] == curr_city])
#         # succ_fcity = road_segments[road_segments['first_city'] == curr_city][['second_city', 'length', 'speed_limit', 'highway']].values.tolist()
#         # succ_scity = road_segments[road_segments['second_city'] == curr_city][['first_city', 'length', 'speed_limit', 'highway']].values.tolist()
#         return self.road_segments[self.road_segments['first_city'] == curr_city][
#                    ['second_city', 'length', 'speed_limit', 'highway']].values.tolist() \
#                + self.road_segments[self.road_segments['second_city'] == curr_city][
#                    ['first_city', 'length', 'speed_limit', 'highway']].values.tolist()


# def read_files(citygps_file_location, road_segments_file_location):
#     """
#     Read txt files and return Pandas DataFrame
#     """
#     citygps = pd.read_csv(citygps_file_location, delimiter=" ", header=None)
#     citygps.columns = ['City', 'la', 'lo']
#     road_segments = pd.read_csv(road_segments_file_location, delimiter=" ", header=None)
#     road_segments.columns = ['first_city', 'second_city', 'length', 'speed_limit', 'highway']
#     return citygps, road_segments


def open_file_gps(input_file):
    data = []
    file = open(input_file, "r")
    for entry in file:
        s = entry.split()
        data.append([s[0], float(s[1]), float(s[2])])
    return data


def open_file_segments(input_file):
    data = []
    file = open(input_file, "r")
    for entry in file:
        s = entry.split()
        data.append([s[0], s[1], float(s[2]), int(s[3]), s[4]])
        data.append([s[1], s[0], float(s[2]), int(s[3]), s[4]])
    return data

def dist_heuristic_bylist(citygps, succ_city, goal_city):
    try:
        # return np.sqrt(np.square(citygps[succ_city][0] - citygps[goal_city][0]) + np.square(citygps[succ_city][1] - citygps[goal_city][1])) # Euclidean distance
        return np.abs(citygps[succ_city][0] - citygps[goal_city][0]) + np.abs(citygps[succ_city][1] - citygps[goal_city][1]) # Mantattan distance
    except KeyError:
        return 0

    # sfound = False
    # gfound = False
    # for city in citygps:
    #     if city[0] == succ_city:
    #         succ_la = city[1]
    #         succ_lo = city[2]
    #         sfound = True
    #
    # # for city in citygps:
    #     if city[0] == goal_city:
    #         goal_la = city[1]
    #         goal_lo = city[2]
    #         gfound = True
    #     if sfound and gfound:
    #         return abs(goal_la - succ_la) + abs(goal_lo - succ_lo)
    # if sfound and gfound:
    #     return np.sqrt(np.square(goal_la - succ_la) + np.square(goal_lo - succ_lo))
    # else:
    #     # Quebec/highway
    #     print('NOT FOUND distance_heuristic')
    return 0

def distance_cost_bylist(segments, curr_city, succ_city):
    print((curr_city, succ_city))
    result = segments[curr_city]
    for r in result:
        if r[0] ==  succ_city:
            # print('+++')
            # print(r)
            return r[1]
    # print(result)
    # return result
    # print('NOT FOUND time dheuristic')
    return 0

def time_cost_bylist(segments, curr_city, succ_city):
    for s in segments:
        if s[0] == succ_city and s[1] == curr_city:
            return s[2]/s[3]
    print('NOT FOUND time cost')
    return 0


def compute_probability(highway):
    """
    Compute probability: 1 in a Million if Interstate (denoted by I-) else 2 in a Million for every mile
    Here we are just returning 1 or 2 based on highway. Will adjust the actual values later in the calculate functions.
    """
    return 1 if 'I-' in highway else 2


def convert_output(city, length, time, prob, route_taken):
    """
        Convert output according to output format.
    """
    return {"total-segments": len(route_taken),
            "total-miles": float(length),
            "total-hours": time,
            "total-expected-accidents": prob / 10 ** 6,
            "route-taken": route_taken}

def convert_output2(city, length, time, prob, route_taken):
    """
        Convert output according to output format
    """
    return {"total-segments": len(route_taken),
            "total-miles": float(length),
            "total-hours": time,
            "total-expected-accidents": prob,
            "route-taken": route_taken}

def successors(start, segments_data, visited):
    """
        Return succesors using segments_data of type = list.
    """
    suc_list = []
    for segment in segments_data:
        if start == segment[0] and segment[1] not in visited:
            suc_list.append([segment[1], segment[2], segment[3], segment[4]])
    # print(suc_list)
    return suc_list

def successors_dict(start, segments_data, visited=None):
    """
        Return succesors using segments_data of type = dict.
        Was added to optimize the program such that
        this function finds out successors using a dict, which is a faster method than using list.
    """
    return segments_data[start]


def cal_distance(start, end, segments_data, citygps):
    """
        Calculate the path from 'start' to 'end' by optimizing distance.
    """
    visited = set([])
    initial_fringe = (0, (0, (start, 0, 0, 0), []))  # FRINGE: heuristic, (cost, city, path) -> city w/ city name, length, speed_limit, highway
    fringe = [initial_fringe]
    heapq.heapify(fringe)
    # print (c.road_segments)
    while fringe:
        _, (curr_cost, (curr_city, curr_length, curr_time, curr_prob), curr_path) = heapq.heappop(fringe)
        # visited.append(curr_city)
        # visited  = list(set(visited))
        visited.add(curr_city)
        # print(visited)
        # i += 1
        # if i == 3:
        #     return 0
        # heu_precompute = dist_heuristic_bylist(citygps, curr_city, end)
        for succ_city_info in successors_dict(start =curr_city, segments_data = segments_data, visited = visited):

            if len(succ_city_info) != 0 and succ_city_info[0] not in visited:

                # if '&' in succ_city_info[0]:
                #     print(succ_city_info[0])
                #     continue
                if succ_city_info[0] == end:  # is goal?
                    # add next step
                    return convert_output(succ_city_info[0],curr_length + succ_city_info[1],\
                                          curr_time + succ_city_info[1] / succ_city_info[2],\
                                          curr_prob + (succ_city_info[1])*compute_probability(succ_city_info[3]), \
                                          deepcopy(curr_path) + [((succ_city_info[0],\
                                                succ_city_info[3] + " for " + str(succ_city_info[1]) + " miles"))] )

                # 'first_city', 'length', 'speed_limit', 'highway'

                # succ_cost = curr_cost + distance_cost_bylist(segments_data,curr_city, succ_city_info[0])  # cost
                # heu_precompute = dist_heuristic_bylist(citygps, succ_city_info[0], end)
                new_path = deepcopy(curr_path) + [
                    ((succ_city_info[0], succ_city_info[3] + " for " + str(succ_city_info[1]) + " miles"))]
                # print(new_path)
                # print('end:', end)
                # dist_heuristic_bylist(citygps, succ_city_info[0], end)
                succ_cost = curr_cost + succ_city_info[1]
                heapq.heappush(fringe, \
                               (succ_cost + dist_heuristic_bylist(citygps, succ_city_info[0], end), (succ_cost, \
                                                                                           (succ_city_info[0],
                                                                                            curr_length +
                                                                                            succ_city_info[1],
                                                                                            curr_time + succ_city_info[
                                                                                                1] / succ_city_info[2],
                                                                                            curr_prob + (succ_city_info[
                                                                                                1]) * compute_probability(
                                                                                                    succ_city_info[3])
                                                                                            ), \
                                                                                           new_path)))
                # curr_prob + (succ_city_info[1]) * compute_probability(
                #     succ_city_info[3])

    return ""


def cal_segments(start, end, segments_data):
    """
    Calculate the path from 'start' to 'end' by optimizing segments.
    """
    visited = set([])
    time = 0
    accident = 0.0
    # segments = 0
    miles = []
    mile = []
    initial = (0, start, [], 0, time,accident)
    subS = "I-"

    seg_queue = [initial]
    heapq.heapify(seg_queue)

    while seg_queue:
        (segment, current, route, t_miles, time, accident) = heapq.heappop(seg_queue)
        # print(type(t_miles))
        next_states = successors(current, segments_data, visited)
        for city in next_states:
            visited.add(city[0])
            t1 = (city[1] / city[2])
            mile = miles.append(city[1])
            if subS in city[3]:
                b = 0.000001 * city[1]
            else:
                b = 0.000002 * city[1]
            if city[0] == end:
                return convert_output2(city[0], t_miles + 1, time + t1, accident + b,deepcopy(route) + [
                    ((city[0], city[3] + " for " + str(city[1]) + " miles"))])

            else:
                new_route = deepcopy(route) + [((city[0], city[3] + " for " + str(city[1]) + " miles"))]
                heapq.heappush(seg_queue, (
                segment + 1, city[0], new_route, t_miles + city[1], time + t1, accident+b))
    return 0


def cal_time(start, end, segments_data, citygps = None):
    """
        Calculate the path from 'start' to 'end' by optimizing time.
    """
    visited = set([])
    initial_fringe = (0, (0, (start, 0, 0, 0),[]))  # FRINGE: heuristic, (cost, city, path) -> city w/ city name, length, speed_limit, highway
    fringe = [initial_fringe]
    heapq.heapify(fringe)
    # print (c.road_segments)
    while fringe:
        _, (curr_cost, (curr_city, curr_length, curr_time, curr_prob), curr_path) = heapq.heappop(fringe)
        visited.add(curr_city)
        # print(visited)
        # print((curr_city, curr_cost))
        # if succ_city_info[0] in
        for succ_city_info in successors_dict( curr_city, segments_data , visited ):
            if len(succ_city_info) != 0 and succ_city_info[0] not in visited:
                if succ_city_info[0] == end:  # is goal?
                    # add next step
                    return convert_output(succ_city_info[0],curr_length + succ_city_info[1],\
                                          curr_time + succ_city_info[1] / succ_city_info[2],\
                                          curr_prob + ( succ_city_info[1])*compute_probability(succ_city_info[3]), deepcopy(curr_path) + [
                    ((succ_city_info[0], succ_city_info[3] + " for " + str(succ_city_info[1]) + " miles"))])

                succ_cost = curr_cost +  succ_city_info[1]/succ_city_info[2]  # cost
                new_path = deepcopy(curr_path) + [
                    ((succ_city_info[0], succ_city_info[3] + " for " + str(succ_city_info[1]) + " miles"))]
                # print(new_path)
                # time_cost_bylist(segments_data, curr_city, succ_city_info[0], end)
                heapq.heappush(fringe,( succ_cost,\
                                       (succ_cost,(succ_city_info[0],curr_length + succ_city_info[1], curr_time + succ_city_info[
                                                                                                1] / succ_city_info[2],
                                                                                            curr_prob + ( succ_city_info[1])*compute_probability(
                                                                                                succ_city_info[3])), \
                                                                                           new_path)))
    return ""


def cal_safe(start, end, segments_data):
    """
        Calculate the path from 'start' to 'end' by optimizing the probability of accidents.
    """
    visited = set([])
    time = 0
    route = []
    accident = 0.0
    initial = (accident, start, [], 0, time, 0)
    subS = "I-"
    seg_queue = [initial]
    heapq.heapify(seg_queue)

    while seg_queue:
        (accident, current, route, t_miles, time, segment) = heapq.heappop(seg_queue)
        next_states = successors(current, segments_data, visited)
        for city in next_states:
            visited.add(city[0])
            t1 = (city[1] / city[2])
            if subS in city[3]:
                b = 0.000001 * city[1]
            else:
                b = 0.000002 * city[1]
            # route.append((city[0],city[3] + " for " + str(city[1])))
            # new = route
            if city[0] == end:
                return convert_output2(city[0],t_miles + city[1],time + t1,accident+b,deepcopy(route) + [
                    ((city[0], city[3] + " for " + str(city[1]) + " miles"))])


            else:
                new_route = deepcopy(route) + [
                    ((city[0], city[3] + " for " + str(city[1]) + " miles"))]
                heapq.heappush(seg_queue, (
                accident + b, city[0], new_route, t_miles + city[1],
                time + t1, segment + 1))
    return 0


def get_route(start, end, cost):
    """
    Find shortest driving route between start city and end city
    based on a cost function.

    1. Your function should return a dictionary having the following keys:
        -"route-taken" : a list of pairs of the form (next-stop, segment-info), where
           next-stop is a string giving the next stop in the route, and segment-info is a free-form
           string containing information about the segment that will be displayed to the user.
           (segment-info is not inspected by the automatic testing program).
        -"total-segments": an integer indicating number of segments in the route-taken
        -"total-miles": a float indicating total number of miles in the route-taken
        -"total-hours": a float indicating total amount of time in the route-taken
        -"total-expected-accidents": a float indicating the expected accident count on the route taken
    2. Do not add any extra parameters to the get_route() function, or it will break our grading and testing code.
    3. Please do not use any global variables, as it may cause the testing code to fail.
    4. You can assume that all test cases will be solvable.
    5. The current code just returns a dummy solution.
    """
    t1 = time.perf_counter_ns()
    gps_data_list = open_file_gps("city-gps.txt")
    t2 = time.perf_counter_ns()
    print("Time gps: ", str((t2 - t1) * 10 ** -(9)))
    segments_data_list = open_file_segments("road-segments.txt")
    t3 = time.perf_counter_ns()
    print("Time segments_data: ", str((t3 - t2) * 10 ** -(9)))
    gps_data = {g[0]: (g[1], g[2]) for g in gps_data_list}
    # count =0
    # for g in gps_data.keys():
    #     count +=1
    #     print((g,':',gps_data[g]))
    # print(count)
    # return 0
    # segments_data = {s[0]: {s[1]: (s[2], s[3], s[4])} for s in segments_data_list}
    segments_data={}
    for s in segments_data_list:
        if s[0] in segments_data.keys():
            segments_data[s[0]].append([s[1],s[2], s[3], s[4]])
        else:
            segments_data[s[0]] = [[s[1],s[2], s[3], s[4]]]
    # print(segments_data)

    if cost == 'segments':
        R = cal_segments(start, end, segments_data_list)
    elif cost == 'distance':
        R = cal_distance(start, end, segments_data, gps_data)
    elif cost == 'time':
        R = cal_time(start, end, segments_data)
    elif cost == 'safe':
        R = cal_safe(start, end, segments_data_list)
    else:
        print("Cost function is invalid")

    # route_taken = [("Martinsville,_Indiana", "IN_37 for 19 miles"),
    #                ("Jct_I-465_&_IN_37_S,_Indiana", "IN_37 for 25 miles"),
    #                ("Indianapolis,_Indiana", "IN_37 for 7 miles")]

    return R

# Please don't modify anything below this line
#
if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise (Exception("Error: expected 3 arguments"))

    (_, start_city, end_city, cost_function) = sys.argv
    if cost_function not in ("segments", "distance", "time", "safe"):
        raise (Exception("Error: invalid cost function"))
    t1 = time.perf_counter_ns()
    result = get_route(start_city, end_city, cost_function)
    t2 = time.perf_counter_ns()
    print("Time: ", str((t2-t1)*10**-(9)))
    if result != None:
        print("Start in %s" % start_city)
        for step in result["route-taken"]:
            print("   Then go to %s via %s" % step)
    else:
        print("No result")
    # Pretty print the route
    # print("Start in %s" % start_city)
    # for step in result["route-taken"]:
    #     print("   Then go to %s via %s" % step)

    print("\n Total segments: %6d" % result["total-segments"])
    print("    Total miles: %10.3f" % result["total-miles"])
    print("    Total hours: %10.3f" % result["total-hours"])
    print("Total accidents: %15.8f" % result["total-expected-accidents"])

    # dataframes = read_files("city-gps.txt", "road-segments.txt")
    #
    # # print(dataframes[0].info())
    # # print(dataframes[0].shape)
    # # print('---')
    # # print(dataframes[1].info())
    # # print(dataframes[1].shape)
    # c = Cities(dataframes[0], dataframes[1])
    # print(cal_distance('start_city', 'goal_city', c))
