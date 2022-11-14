from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ProblemInstanceG:
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
	
	def draw2d(self, buses=None, draw_lines=True):
		xy = np.array([node[0] for node in self.nodes])
		colors = ["red" for node in self.nodes]
		for depot in self.depots:
			colors[depot] = "black"
		#end for depot
		
		jibun_wo = ["blue","red","orange","green","purple"]
		
		if buses is not None:
			no_of_buses = len(buses)
			
			for i in range(no_of_buses):
				for node in buses[i][1:]:
					colors[node] = jibun_wo[i%5]
				#end for node
				buses[i].append(buses[i][0])
			#end for i
		#end if
		
		plt.scatter(xy[1:,0], xy[1:,1], c=colors[1:], s=20)
		if buses is not None and draw_lines:
			for i in range(len(buses)):
				for j in range(len(buses[i])-1):
					t1 = self.nodes[buses[i][j]][0]
					t2 = self.nodes[buses[i][j+1]][0]
					plt.plot(
						[t1[0],t2[0]], [t1[1],t2[1]],
						color=jibun_wo[i%5], linestyle="--"
					)
				#end for i
			#end for bus
		#end if
		plt.grid(linestyle="--", zorder=-1)
		
		for j in range(1, self.N+1):
			plt.annotate(j, (xy[j,0], xy[j,1]))
		#end for i
		plt.show()
	#end def
	
	def greedy(self):
		# we need the index of every node so we can access them later
		nodes = list(zip(range(self.N+1), self.nodes))[1:]
		
		starting_node = nodes[0] #depot
		nodes = nodes[1:] #we don't need the depot info anymore
		# each vehicle will have its own list of available nodes
		nodes = [nodes for _ in range(self.K)]
		
		# K vehicles available
		# we need the list of nodes, value, current capacity and node counter
		#  for each one
		buses = [[] for _ in range(self.K)]
		values = [0 for _ in range(self.K)]
		capacities = [self.capacity for _ in range(self.K)]
		counters = [0 for _ in range(self.K)]
		full = [False for _ in range(self.K)]
		conv = True #does the algorithm return a feasible solution?
		
		# set the current location of each vehicle to the depot
		for j in range(self.K):
			buses[j].append(starting_node[0])
		#end for j
		
		# ALGORITHM START
		# we need to have a total of N-1 nodes inside all vehicles
		while sum(counters) < self.N - 1:
			# first, we suppose that the best feasible node doesn't exist
			next_node = None
			next_node_dist = float("inf")
			next_j = None
			
			# find the best next node for each vehicle
			for j in range(self.K):
				# if the vehicle is full or doesn't have enough space for any of the
				#  remaining nodes
				if capacities[j]==0 or full[j]:
					full[j] = True
					continue
				#end if
				
				last_index = buses[j][-1] #current location of jth vehicle
				# suppose the best feasible node for the current vehicle doesn't exist
				candidate = None
				candidate_capacity = self.capacity + 1
				candidate_dist = float("inf")
				
				# find a node that can fit inside the jth vehicle
				while capacities[j] < candidate_capacity:
					# no node can fit inside
					if len(nodes[j]) < 1:
						full[j] = True
						break
					#end if
					
					# find the index of the closest node
					index_min_j = min(range(len(nodes[j])), key=lambda i:
						self.metric(
							self.nodes[last_index][0],
							nodes[j][i][1][0]
						)
					)
					
					candidate = nodes[j][index_min_j]
					candidate_capacity = candidate[1][1]
					
					# check if the closest node fits inside the jth vehicle
					if capacities[j] < candidate_capacity:
						# if doesn't fit in now, it won't fit in later -> remove it
						nodes[j].remove(candidate)
					else:
						# we found the best one that fits inside the jth vehicle
						candidate_dist = self.metric(
							self.nodes[last_index][0],
							candidate[1][0]
						)
					#end if
				#end while capacities[j]
				
				# now, out of all the feasible ones, find the best one
				if candidate_dist < next_node_dist:
					next_node = candidate
					next_node_dist = candidate_dist
					next_j = j
				#end if
			#end for j
			
			# if we haven't found a single feasible node
			if next_node is None:
				# the solution is not feasible; STOP
				conv = False
				break
			#end if
			
			# we found our best candidate; remove it from each list of unvisited nodes
			for j in range(self.K):
				# each node has a unique position on the map
				# by using list comprehension, we don't have to check
				#  if the node is even inside the list
				nodes[j] = [node for node in nodes[j] if node != next_node]
			#end for j
			
			# finally, add the node to the corresponding list and update the lists
			buses[next_j].append(next_node[0])
			values[next_j] += next_node_dist
			capacities[next_j] -= next_node[1][1]
			counters[next_j] += 1
		#end while sum(counters)
		
		# each vehicle still needs to return to where it started
		for j in range(self.K):
			# check if we even used the vehicle
			if len(buses[j]) > 1:
				values[j] += self.metric(
					self.nodes[buses[j][-1]][0],
					self.nodes[buses[j][0]][0]
				)
			#end if
		#end for j
		
		return buses, values, capacities, conv
	#end def
	
	def printSolution(self, buses, values, conv, solutionFilePath=None, logsFilePath=None):
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
		
		for j, bus in enumerate(buses):
			bus_out = ", ".join([str(i) for i in bus[1:]])
			line_out = "Vozilo #" + str(j+1) + ": " + bus_out
			print(line_out)
			log_out += line_out + "\n"
		#end for j, bus
		
		x_star = int(sum(values))
		line_out = "Trošak " + str(x_star)
		print(line_out)
		log_out += line_out + "\n"
		sol_out += str(x_star) + " "
		
		line_out = "Dopustivo: " + str(conv)
		print(line_out)
		log_out += line_out + "\n"
		
		print()
		log_out += "\n"
		
		if log_flag:
			logs_file.write(log_out)
			logs_file.close()
		#end if
		
		x = self.optimal_solution
		aps = abs(x - x_star)
		sol_out += str(aps) + " "
		rel = aps/(abs(x))
		sol_out += str(rel) + " "
		
		solution_bool = ""
		if conv:
			if x_star <= x:
				solution_bool = "DA"
			else:
				solution_bool = "NE"
			#end if
		else:
			solution_bool = "NIJE_DOPUSTIVO"
		#end if
		
		sol_out += solution_bool + "\n"
		
		if sol_flag:
			solution_file.write(sol_out)
			solution_file.close()
		#end if
	#end def
#end class definition


if __name__ == "__main__":
	
	primjer = ProblemInstanceG("./instance/A-n32-k5.vrp")
	
	buses, values, capacities, conv = primjer.greedy()
	"""
	print(buses)
	print(values)
	print(capacities)
	print("konvergira:", conv)
	print(sum(values))
	print(primjer.optimal_solution, end="\n\n")
	
	primjer.draw2d(buses)
	"""
	solutions_folder = "./rjesenja/"
	logs_folder = "./logs/"
	
	primjer.printSolution(buses, values, conv)
	
	"""
	primjer = ProblemInstance("./instance/A-n32-k5.vrp")
	print(primjer.optimal_solution)
	"""
#end main
