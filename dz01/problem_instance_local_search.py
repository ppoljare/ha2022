from math import sqrt
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class ProblemInstanceLS:
	__slots__ = [
		"path", "name", "problemType", "N", "K", "metric",\
		"capacity", "nodes", "depots", "optimal_solution"
	]
	
	def __init__(self, path, optimalSolutionsPath="./opisInstanci.xlsx"):
		self.path = path
		self.nodes = []
		self.depots = []
		
		# Parse test case file
		file = open(self.path, 'r')
		section = "neutral"
		
		for line in file:
			line = line.strip()
			
			if line=="EOF":
				break
			elif line=="NODE_COORD_SECTION":
				section = "NODE_COORD"
				continue
			elif line=="DEMAND_SECTION":
				section = "DEMAND"
				continue
			elif line=="DEPOT_SECTION":
				section = "DEPOT"
				continue
			#end if
			
			if section=="NODE_COORD":
				line = line.split(" ")
				if len(line)==1:
					line = line[0].split("\t")
				#end if
				k = int(line[0])
				coordinates = [int(x_i) for x_i in line[1:]]
				pos = tuple(coordinates)
				self.nodes[k] = [pos, 0]
				continue
			elif section=="DEMAND":
				line = line.split(" ")
				if len(line)==1:
					line = line[0].split("\t")
				#end if
				k = int(line[0])
				self.nodes[k][1] = int(line[1])
				continue
			elif section=="DEPOT":
				if line=="-1":
					section = "neutral"
				else:
					k = int(line)
					self.depots.append(k)
				continue
			#end if
			
			line = line.split(" : ")
			ltype, lval = line[:2]
			
			if ltype=="NAME":
				self.name = lval.strip()
				self.K = int(lval[lval.rfind('k')+1:])
			elif ltype=="TYPE":
				self.problemType = lval
			elif ltype=="DIMENSION":
				self.N = int(lval)
				self.nodes = [[(0,0),0] for _ in range(self.N+1)]
			elif ltype=="EDGE_WEIGHT_TYPE":
				if lval=="EUC_3D":
					self.metric = lambda x, y: sqrt(
						(x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2
					)
				else:
					# we will use 2D-Euclid metric as the "default" one
					self.metric = lambda x, y: sqrt(
						(x[0]-y[0])**2 + (x[1]-y[1])**2
					)
				#end if
			elif ltype=="CAPACITY":
				self.capacity = int(lval)
			#end if
		#end for line
		file.close()
		
		# Find the current optimal solution for this instance
		optimalSolutionsData = pd.read_excel(optimalSolutionsPath)
		data_list = optimalSolutionsData.to_dict("records")
		n = len(data_list)
		indices = [i for i in range(n) if data_list[i]["Naziv_instance"]==self.name]
		self.optimal_solution = data_list[indices[0]]["UB"]
	#end def
	
	def draw2d(self, solution=None, draw_lines=True):
		xy = np.array([node[0] for node in self.nodes])
		colors = ["red" for node in self.nodes]
		for depot in self.depots:
			colors[depot] = "black"
		#end for depot
		
		jibun_wo = ["blue","red","orange","green","purple"]
		allDepots = []
		allCities = []
		n = 0
		s = []
		
		if solution is not None:
			s = solution.copy()
			n = len(solution)
			j = 0
			allDepots = [i for i in solution if i in self.depots]
			allDepots.append(allDepots[0])
			
			for i in range(self.K):
				j += 1
				
				while j < n and solution[j] != allDepots[i]:
					current_node = solution[j]
					colors[current_node] = jibun_wo[i%5]
					j += 1
				#end while j
			#end for i
		#end if
		
		plt.scatter(xy[1:,0], xy[1:,1], c=colors[1:], s=20)
		if solution is not None and draw_lines:
			for j in range(n-1):
				t1 = self.nodes[solution[j]][0]
				t2 = self.nodes[solution[j+1]][0]
				color = colors[solution[j]]
				if solution[j] in self.depots:
					color = colors[solution[j+1]]
				#end if
				plt.plot(
					[t1[0],t2[0]], [t1[1],t2[1]],
					color=color, linestyle="--"
				)
			#end for j
			
			t1 = self.nodes[solution[-1]][0]
			t2 = self.nodes[solution[0]][0]
			color = colors[solution[-1]]
			plt.plot(
				[t1[0],t2[0]], [t1[1],t2[1]],
				color=color, linestyle="--"
			)
		#end if
		plt.grid(linestyle="--", zorder=-1)
		
		for j in range(1, self.N+1):
			plt.annotate(j, (xy[j,0], xy[j,1]))
		#end for i
		plt.show()
	#end def
	
	def objectiveFunction(self, solution):
		# check if the solution is even feasible
		# we add this here to speed up the search for the best neighbour
		if not self.isFeasible(solution):
			return float("inf")
		#end if
		
		s = solution.copy()
		n = len(s) + 1
		res = 0
		
		allDepots = [i for i in s if i in self.depots]
		allDepots.append(allDepots[-1])
		s.append(allDepots[-1])
		depot_index = 0
		
		# go through all the nodes
		for j in range(1, n):
			# get the indices of the current node and the next
			curr_j = s[j-1]
			next_j = s[j]
			
			# if we found a depot -> calculate the distance to the previous one
			if s[j]==allDepots[depot_index+1]:
				next_j = allDepots[depot_index]
				depot_index += 1
			#end if
			
			# calculate distance from current node to the next
			res += self.metric(self.nodes[curr_j][0], self.nodes[next_j][0])
		#end for j
		
		return res
	#end def
	
	def isFeasible(self, solution):
		# check if the first node is actually a depot
		if solution[0] not in self.depots:
			return False
		#end if
		
		# check if we have the right amount of depots in the solution
		allDepots = [i for i in solution if i in self.depots]
		
		if len(allDepots) != self.K:
			return False
		#end if
		
		# check if we went through all the cities
		allCities = sorted([i for i in solution if i not in self.depots])
		allCitiesReq = [i for i in range(1, self.N+1) if i not in self.depots]
		
		if allCities != allCitiesReq:
			return False
		#end if
		
		# move the starting depot to the end
		allDepots.append(allDepots[0])
		allDepots = allDepots[1:]
		# it doesn't really matter which depot is at the end,
		#  we just need to shift all depots to the left and add one to the end
		
		n = len(solution)
		j = 0 #current node
		
		# check if all vehicles have enough capacity for this route
		for bus_index in range(self.K):
			current_capacity = self.capacity
			j += 1 #we use this to skip the starting depot of current vehicle
			
			while j < n and solution[j] != allDepots[bus_index]:
				# get the index of current node
				current_node = solution[j]
				
				# check if the current vehicle has enough space for the current node
				current_capacity -= self.nodes[current_node][1]
				if current_capacity < 0:
					# we don't have enough space -> NOT FEASIBLE
					return False
				#end if
				
				j += 1
			#end while j
		#end for bus
		
		# we had enough space for every node -> FEASIBLE
		return True
	#end def
	
	def getNeighboursSwitch(self, solution, max_distance=0):
		"""
		This function generates a list of neighbours by creating a list
		of pairs (i,j) and then switching the ith and the jth element
		of the solution for each one of the pairs.
		By setting the "max_distance" parameter to a positive integer,
		we can prevent the function from moving the elements of the solution
		further than "max_distance" steps to the left or right.
		"""
		neighbours = []
		n = len(solution)
		
		# max_distance == 0 -> get all neighbours
		# max_distance > 0  -> get closest "max_distance" neighbours
		if max_distance < 1:
			max_distance = n
		#end def
		
		for i in range(1, n):
			m = min(n, i+max_distance+1)
			for j in range(i+1, m):
				neighbour = solution.copy()
				neighbour[i] = solution[j]
				neighbour[j] = solution[i]
				neighbours.append(neighbour)
			#end for j
		#end for i
		
		return neighbours
	#end def
	
	def getNeighboursShift(self, solution, max_distance=10):
		"""
		This function generates a list of neighbours by shifting the entire
		solution to the left n-1 times, where n is the length of the solution.
		There are two different ways this function can work, denoted by the
		"shiftMode" parameter:
		By setting the "max_distance" parameter to a positive integer,
		we can prevent the function from moving the elements of the solution
		further than "max_distance" steps to the left or right.
		"""
		neighbours = []
		n = len(solution)
		
		for i in range(1, max_distance):
			for j in range(0, n-1):
				neighbour = solution[1:]
				toInsert = neighbour.pop(j)
				neighbour = [solution[0]] + neighbour[i:] + neighbour[:i]
				neighbour.insert(j+1, toInsert)
				neighbours.append(neighbour)
			#end for j
		#end for i
		
		return neighbours
	#end def
	
	def getBestNeighbour(self, neighbours, searchMode="all"):
		bestRouteLength = self.objectiveFunction(neighbours[0])
		bestNeighbour = neighbours[0]
		
		for neighbour in neighbours:
			currentRouteLength = self.objectiveFunction(neighbour)
			if currentRouteLength < bestRouteLength:
				bestRouteLength = currentRouteLength
				bestNeighbour = neighbour
				
				if searchMode == "firstBest":
					return bestNeighbour, bestRouteLength
				#end if
			#end if
		#end for neighbour
		
		return bestNeighbour, bestRouteLength
	#end def
	
	def getRandomSolution(self):
		# we need a list of all citites
		nodes = [i for i in range(1, self.N+1) if i not in self.depots]
		solution = [] #starting solution
		# we need exactly K depots in the solution
		for k in range(self.K):
			trash = [] #empty the trash
			# pick a random starting depot
			depot = random.choice(self.depots)
			solution.append(depot)
			# we will try to construct a feasible solution
			capacity = self.capacity
			while capacity > 0 and len(nodes) > 0:
				# pick a non-depot node randomly
				next_node = random.choice(nodes)
				# if the current vehicle has enough space for the next node
				if self.nodes[next_node][1] <= capacity:
					# add it to the solution
					#  and deduct its weight from the current capacity
					solution.append(next_node)
					capacity -= self.nodes[next_node][1]
				else:
					# otherwise, move it to trash, sort of our "to-do list"
					trash.append(next_node)
				#end if
				# wheter or not the node fits inside the current vehicle,
				#  remove it from the list because we already visited it
				nodes.remove(next_node)
			#end while
			# add all nodes from the trash back into the list of nodes
			nodes += trash
		#end for k
		
		return solution #fingers crossed that this is a feasible solution
	#end def
	
	def localSearch(self, neighbourMode=("switch", 0), searchMode="all", max_inits=20):
		# initialization
		currentSolution = self.getRandomSolution()
		counter = 1 #we already tried once
		
		# try to generate a feasible solution
		while not self.isFeasible(currentSolution):
			if counter >= max_inits:
				# we tried too many times, return the one you've got,
				#  even though it's not feasible
				return currentSolution, self.objectiveFunction(currentSolution)
			#end if
			currentSolution = self.getRandomSolution()
			counter += 1
		#end while
		
		currentRouteLength = self.objectiveFunction(currentSolution)
		neighbours = []
		
		# the first parameter of "neighbourMode" denotes the type
		if neighbourMode[0] == "switch":
			# the second parameter denotes the max_distance
			neighbours = self.getNeighboursSwitch(
				currentSolution, neighbourMode[1]
			)
		else:
			# the second parameter denotes if we're keeping the first depot,
			#  keeping them all, or we're generating them both
			neighbours = self.getNeighboursShift(
				currentSolution, neighbourMode[1]
			)
		#end if
		
		bestNeighbour, bestRouteLength = self.getBestNeighbour(
			neighbours, searchMode
		)
		
		while bestRouteLength < currentRouteLength:
			currentSolution = bestNeighbour
			currentRouteLength = bestRouteLength
			
			if neighbourMode[0] == "switch":
				neighbours = self.getNeighboursSwitch(
					currentSolution, neighbourMode[1]
				)
			else:
				neighbours = self.getNeighboursShift(
					currentSolution, neighbourMode[1]
				)
			#end if
			
			bestNeighbour, bestRouteLength = self.getBestNeighbour(
				neighbours, searchMode
			)
		#end while
		
		return currentSolution, currentRouteLength
	#end def
	
	def multistart(self, no_of_starts=10, neighbourMode=("switch", 0), searchMode="all", max_inits=20):
		# we need a starting solution for comparison
		best = self.getRandomSolution()
		best_eval = self.objectiveFunction(best)
		
		for n in range(no_of_starts):
			solution, solution_eval = self.localSearch(
				neighbourMode=neighbourMode,
				searchMode=searchMode,
				max_inits=max_inits
			)
			
			if solution_eval < best_eval:
				best, best_eval = solution, solution_eval
			#end if
		#end for n
		
		return best, best_eval
	#end def
	
	def printSolution(self, solution, solutionFilePath=None, logsFilePath=None):
		sol_flag = False
		sol_out = ""
		
		log_flag = False
		log_out = ""
		
		if solutionFilePath is not None:
			solution_file = open(solutionFilePath, 'a')
			sol_flag = True
		#end if
		
		if logsFilePath is not None:
			logs_file = open(logsFilePath, 'a')
			log_flag = True
		#end if
		
		print(self.name)
		log_out += self.name + "\n"
		sol_out += self.name + " "
		
		allDepots = [i for i in solution if i in self.depots]
		allDepots.append(allDepots[0])
		allDepots = allDepots[1:]
		n = len(solution)
		j = 0 #current node
		
		for bus_index in range(self.K):
			j += 1 #we use this to skip the starting depot of current vehicle
			bus_out = []
			while j < n and solution[j] != allDepots[bus_index]:
				bus_out.append(solution[j])
				j += 1
			#end while j
			bus_out = ", ".join([str(i) for i in bus_out])
			line_out = "Vozilo #" + str(bus_index+1) + ": " + bus_out
			print(line_out)
			log_out += line_out + "\n"
		#end for bus_index
		
		x = self.optimal_solution
		x_star = float("inf")
		aps = float("inf")
		rel = float("inf")
		solution_bool = ""
		
		conv = False
		f_solution = self.objectiveFunction(solution)
		
		if f_solution < float("inf"):
			x_star = int(f_solution)
			conv = True
			aps = abs(x - x_star)
			rel = aps/(abs(x))
			
			if x_star <= x:
				solution_bool = "DA"
			else:
				solution_bool = "NE"
			#end if
		else:
			solution_bool = "NIJE_DOPUSTIVO"
		#end if
		
		line_out = "Trošak " + str(x_star)
		print(line_out)
		log_out += line_out + "\n"
		sol_out += str(x_star) + " "
		
		#line_out = "Dopustivo: " + str(conv)
		#print(line_out)
		#log_out += line_out + "\n"
		
		print()
		log_out += "\n"
		
		if log_flag:
			logs_file.write(log_out)
			logs_file.close()
		#end if
		
		sol_out += str(aps) + " " + str(rel) + " "
		sol_out += solution_bool + "\n"
		
		if sol_flag:
			solution_file.write(sol_out)
			solution_file.close()
		#end if
	#end def
#end class definition


if __name__ == "__main__":
	
	instances_folder = "./instance/"
	files_list = sorted(os.listdir(instances_folder))
	
	max_inits = 100
	"""
	for file_name in files_list:
		instance = ProblemInstanceLS(instances_folder+file_name)
		startingSolution = None
		solution_eval = float("inf")
		for i in range(max_inits):
			startingSolution = instance.getRandomSolution()
			if instance.isFeasible(startingSolution):
				solution_eval = instance.objectiveFunction(startingSolution)
				print("iteracija:", i)
				break
			#end if
		#end for
		#print(startingSolution)
		print(solution_eval)
		print()
	#end for file_name
	"""
	"""
	file_name = "A-n33-k5.vrp"
	
	instance = ProblemInstanceLS(instances_folder+file_name)
	startingSolution = None
	solution_eval = float("inf")
	for i in range(max_inits):
		startingSolution = instance.getRandomSolution()
		if instance.isFeasible(startingSolution):
			solution_eval = instance.objectiveFunction(startingSolution)
			print("iteracija:", i)
			break
		#end if
	#end for
	"""
	instance_name = "A-n32-k5"
	#instance_name = "X-n125-k30"
	
	primjer = ProblemInstanceLS("./instance/"+instance_name+".vrp")
	
	"""
	solution, solution_eval = primjer.multistart(
		no_of_starts=10, 
		neighbourMode=("switch", 50), 
		searchMode="all", 
		max_inits=20
	)
	"""
	#start = primjer.getRandomSolution()
	"""
	solution = [1, 11, 25, 7, 4, 10, 3, 32, 23, 1, 17, 28, 21, 31, 20, 15, 24, 19, 30, 27, 1, 5, 2, 8, 9, 22, 16, 1, 6, 18, 12, 26, 13, 29, 1, 14]
	print(solution)
	
	i = 3
	j = 5
	
	neighbour = solution[1:]
	print(neighbour)
	print()
	neighbour.pop(j)
	print(neighbour[i:])
	print(neighbour[:i])
	print()
	neighbour = [solution[0]] + neighbour[i:] + neighbour[:i]
	neighbour.insert(j+1, solution[j+1])
	print(neighbour)
	"""
	
	#N = primjer.getNeighboursShift(start)
	#for i in range(10):
	#	print(N[i][:10])
	#end for i
	
	#print(solution)
	#print(solution_eval)
	#primjer.draw2d(solution)
#end main
