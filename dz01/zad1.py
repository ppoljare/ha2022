import os
import numpy as np
import matplotlib.pyplot as plt
from problem_instance_greedy import ProblemInstanceG

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
	solutions_path = solutions_folder + "rjesenjaGreedy.txt"
	logs_path = logs_folder + "logsGreedy.txt"
	
	for file_name in files_list:
		instance = ProblemInstanceG(instances_folder+file_name)
		buses, values, capacities, conv = instance.greedy()
		instance.printSolution(buses, values, conv, solutions_path, logs_path)
	#end for file_name
#end main
