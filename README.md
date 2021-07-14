
# Part 2
## Road Trip!

### Problem Statement:

* The problem was to output a route from a certain 'start city' to 'end city'. This had to be done by optimizing 4 parameters (indivisually).
* The 4 parameters were "distance", "time", "segments" and "safe"
* The goal with using "distance", "time" and "segments" was to get a route with the shortest distance, time taken to reach 'end city' and number of segments.
* The goal with "safe" was to get a route with the least probability of accidents. Routing via an Interstate has a probability of 1 in a Million per mile, otherwise 1 in a Million per mile.


#### Data given:

* Two files were given:
    * city_gps.txt: Contains GPS co-ordinates for the cities in US.
        * Missing values: cities for state 'Quebec' and no highways.
    * road_segments.txt: Contains the paths from one city to another (or a highway to city or vice-versa)
* ELAI_A1_2_Preprocessing.ipynb (Jupyter notebook) was used to analyze and pre-process the given data files.


### Approach to the Problem:

* Our approach to the problem initially was to design heuristics for each of the parameters so that we can get curtail our seach space and reach to the 'end city' computationally faster than normal search algorithms.
* Hence, A* search was implemented.
* For A* search: f(s) = c(s') + h(s) ... [where, c -> cost to reach the current state and h(s) is the heuristic]. Hence, we have to elect the minimal f(s) for any given state s, arriving to state s from s'.
* To implement A* search heapq was used. (Priority queue was an option; but heapq handles conflicts better)
* The the f(s) value would be used to pop the element from the heapq. (Smaller value of f(s) is better)

#### The following approches, specific for a particular parameter, were followed to implement A* search for the problem.

1. Optimization parameter = "distance":
    * For this problem, the distance had to be minimized.
    Cost c(s') = actual distance to reach current city.
    * There were two options that we considered for heuristic calculation i.e. (1) using Manhattan distance and (2) using Euclidean distance.
    * The compupataion for Euclidean distance took longer than Manhattan distance, as it involves squaring and taking the square root. Hence we chose to go with Manhattan distance for h(s).
    * These distances were calculated using the latitude and longitude information in the city_gps.txt data.
    
2. Optimization parameter = "time":
    * For this problem, the time taken had to be minimized.
    * Cost c(s') = actual time taken to reach current city.
    * Several heuristic values were considered such as (1) taking the previous speed limit for the remaining heuristic distance, (2) taking the average speed limit for the remaining heuristic distance and (3) adding the heuristic distance from current city (successor to current in the code) to the 'end city'
    * But the program was performing better without the heuristic, hence h(s) = 0 was chosen.
    
3. Optimization parameter = "segments":
    * For this problem, the number of segments needed to reach the goal (i.e roads intuitively) had to be minimized.
    * For this problem the code was running fast with just f(s) = c(s').
    * Hence we decided to use only the cost, with the penultimate aim to reduce complexity and also to reduce the overall computation time.
    * Finally, the cost-based search approach was followed in this problem.

4. Optimization parameter = "safe":

    * For this problem, the probability of accidents (i.e roads intuitively) had to be minimized.
    * This was also implements using f(s) = c(s'), as we cannot predict the road-way that will minimize the probability of accidents.
    * Hence we opted to go for h(s) = 0 as well.
    
    
### Challenges faced:

* The main challenge was to find an admissible heuristic which optimizes the search.
* Another challenge was to reduce the computation time required to do the search. Following actions were taken for the same:
    - Inirtially Pandas library was being used to read the data files as DataFrames, since we anticipated it would be easier to slice the data. -- - Also class Cities was implemented so that the program wouldn't have to reference the DataFrame object in every function call from it's own arguments, but rather access the DataFrame objects in Cities class.
    - But the computation time with DataFrames was very slow. (~ 15 mins for routing cities 500 miles apart).
    _ This was then changed to lists of lists to store the data (worked with significant improvement) and eventually dictionaies data-structure was usd to yield better results.
    - Even after these chages the program was not giving staisfactory run-times.
    - To further optimize the program, we changed the data-type 'visited' array to set from list, to give much better run-times.
    - The main issue was the visited list being populated again and again, which increased the computation time.
    - The choice of using Manhattan distance as a heuristic was made to reduce run-time as well. Since we were getting the same output for both Mnahattan distance and Euclidean distance.
    
    
    
