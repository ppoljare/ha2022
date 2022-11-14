import os
import numpy as np
import matplotlib.pyplot as plt
from problem_instance_local_search import ProblemInstanceLS

if __name__ == "__main__":
	"""
	primjer = ProblemInstanceG("./instance/A-n32-k5.vrp")
	print(primjer.name)
	print(primjer.problemType)
	print(primjer.N)
	print(primjer.K)
	print(primjer.metric)
	print(primjer.capacity)
	print(primjer.nodes)
	print(primjer.optimal_solution)
	print()
	
	buses, values, capacities, conv = primjer.greedy()
	print(buses)
	print(values)
	print(capacities)
	print("konvergira:", conv)
	
	primjer.draw2d()
	primjer.draw2d(buses)
	#print(nodes)
	"""
	
	instances_folder = "./instance/"
	solutions_folder = "./rjesenja/"
	logs_folder = "./logs/"
	
	files_list = sorted(os.listdir(instances_folder))
	files_list.remove("X-n1001-k43.vrp")
	#files_list.append("X-n1001-k43.vrp")
	
	playlist = [1, 0, 0, 0]
	
	#### a) Neighbourhood: switch (0), Find first best
	if playlist[0] == 1:
		solutions_path = solutions_folder + "rjesenjaLS1_firstBest.txt"
		logs_path = logs_folder + "logsLS1_firstBest.txt"
		
		for file_name in files_list:
			instance = ProblemInstanceLS(instances_folder+file_name)
			
			no_of_starts = 30
			max_distance = 0
			if instance.name[0]=='X':
				no_of_starts = 10
				max_distance = 1
			#end if
			
			bestSolution, bestRouteLength = instance.multistart(
				no_of_starts=no_of_starts, 
				neighbourMode=("switch", max_distance), 
				searchMode="firstBest", 
				max_inits=20
			)
			instance.printSolution(bestSolution, solutions_path, logs_path)
			#instance.printSolution(bestSolution)
		#end for file_name
	#end if
	
	
	#### b) Neighbourhood: switch (0), Find best
	if playlist[1] == 1:
		solutions_path = solutions_folder + "rjesenjaLS1_searchAll.txt"
		logs_path = logs_folder + "logsLS1_searchAll.txt"
		
		for file_name in files_list:
			instance = ProblemInstanceLS(instances_folder+file_name)
			
			no_of_starts = 30
			max_distance = 0
			if instance.name[0]=='X':
				no_of_starts = 10
				max_distance = 1
			#end if
			
			bestSolution, bestRouteLength = instance.multistart(
				no_of_starts=no_of_starts, 
				neighbourMode=("switch", max_distance), 
				searchMode="all", 
				max_inits=20
			)
			instance.printSolution(bestSolution, solutions_path, logs_path)
			#instance.printSolution(bestSolution)
		#end for file_name
	#end if
	
	
	#### c) Neighbourhood: shift (both), Find first best
	if playlist[2] == 1:
		solutions_path = solutions_folder + "rjesenjaLS2_firstBest.txt"
		logs_path = logs_folder + "logsLS2_firstBest.txt"
		
		for file_name in files_list:
			instance = ProblemInstanceLS(instances_folder+file_name)
			
			no_of_starts = 40
			max_distance = 10
			if instance.name[0]=='X':
				no_of_starts = 10
				max_distance = 2
			#end if
			
			bestSolution, bestRouteLength = instance.multistart(
				no_of_starts=no_of_starts, 
				neighbourMode=("shift", max_distance), 
				searchMode="firstBest", 
				max_inits=20
			)
			instance.printSolution(bestSolution, solutions_path, logs_path)
			#instance.printSolution(bestSolution)
		#end for file_name
	#end if
	
	
	#### d) Neighbourhood: shift (both), Find best
	if playlist[3] == 1:
		solutions_path = solutions_folder + "rjesenjaLS2_searchAll.txt"
		logs_path = logs_folder + "logsLS2_searchAll.txt"
		
		for file_name in files_list:
			instance = ProblemInstanceLS(instances_folder+file_name)
			
			no_of_starts = 40
			max_distance = 10
			if instance.name[0]=='X':
				no_of_starts = 10
				max_distance = 2
			#end if
			
			bestSolution, bestRouteLength = instance.multistart(
				no_of_starts=no_of_starts, 
				neighbourMode=("shift", max_distance), 
				searchMode="all", 
				max_inits=20
			)
			instance.printSolution(bestSolution, solutions_path, logs_path)
			#instance.printSolution(bestSolution)
		#end for file_name
	#end if
#end main
