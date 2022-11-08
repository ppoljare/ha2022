from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

class ProblemInstance:
	__slots__ = [
		"path", "name", "problemType", "N", "metric",\
		"capacity", "nodes", "depots", "optimal_solution"
	]
	
	def __init__(self, path, optimPath="./optimal_solutions.txt"):
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
				k = int(line[0])
				coordinates = [int(x_i) for x_i in line[1:]]
				pos = tuple(coordinates)
				self.nodes[k] = [pos, 0]
				continue
			elif section=="DEMAND":
				line = line.split(" ")
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
				self.name = lval
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
		file = open(optimPath, 'r')
		col_names = file.readline()
		col_names = col_names.split("\t")
		name_col = col_names.index("Naziv_instance")
		k_col = col_names.index("K")
		ub_col = col_names.index("UB")
		
		for line in file:
			line = line.split("\t")
			if line[name_col]==self.name:
				self.optimal_solution = (int(line[k_col]), int(line[ub_col]))
				break
			#end if
		#end for line
		file.close()
	#end def
	
	def draw2d(self, buses=None):
		xy = np.array([item[0] for item in self.nodes])
		colors = ["red" for item in self.nodes]
		for depot in self.depots:
			colors[depot] = "black"
		#end for depot
		
		"""routes = [
			[27,8,14,18,20,32,22],
			[31,17,2,13],
			[28,25],
			[21,6,26,11,16,23,10,9,19,30],
			[15,29,12,5,24,4,3,7],
		]"""
		
		jibun_wo = ["blue","red","orange","green","purple"]
		
		demands = [item[1] for item in self.nodes]
		#total_demands = [0 for _ in routes]
		
		
		if buses is not None:
			no_of_buses = len(buses)
			
			for i in range(no_of_buses):
				for node in buses[i][1:]:
					colors[node] = jibun_wo[i]
				#end for node
				buses[i].append(buses[i][0])
			#end for i
		#end if
		
		#prvi_bus = [1, 31, 27, 25, 17, 28, 15, 2]
		
		#for node in prvi_bus:
		#	colors[node] = "orange"
		#end for node
		
		plt.scatter(xy[1:,0], xy[1:,1], c=colors[1:], s=20)
		if buses is not None:
			for i in range(len(buses)):
				for j in range(len(buses[i])-1):
					t1 = self.nodes[buses[i][j]][0]
					t2 = self.nodes[buses[i][j+1]][0]
					plt.plot([t1[0],t2[0]], [t1[1],t2[1]], color=jibun_wo[i], linestyle="--")
				#end for i
			#end for bus
		#end if
		plt.grid(linestyle="--", zorder=-1)
		
		#which_demands = [31,27,17,13,2,8]
		#total_demands = 0
		
		#distances = [self.metric(nodes[self.depots[0]][0], item[0]) for (key, item) in nodes]
		#print(self.depots[0])
		#print(self.nodes[1][0])
		distances = [self.metric(self.nodes[self.depots[0]][0], coord) for (coord, demand) in self.nodes]
		#distances = [self.metric(self.nodes[31][0], coord) for (coord, demand) in self.nodes]
		#distances = [self.metric(self.nodes[27][0], coord) for (coord, demand) in self.nodes]
		#distances = [self.metric(self.nodes[17][0], coord) for (coord, demand) in self.nodes]
		#distances = [self.metric(self.nodes[13][0], coord) for (coord, demand) in self.nodes]
		#distances = [self.metric(self.nodes[2][0], coord) for (coord, demand) in self.nodes]
		#distances = [self.metric(self.nodes[8][0], coord) for (coord, demand) in self.nodes]
		
		"""for i in which_demands:
			total_demands += demands[i]
		#end for i
		print(total_demands)"""
		
		"""for i, demand in enumerate(demands[2:]):
			j = i+2
			#val = round(demand/distances[j], 2)
			#val = demand
			val = round(distances[j], 2)
			plt.annotate((j, val), (xy[j,0], xy[j,1]))
		#end for i, demand"""
		
		#print(total_demands)
		
		
		for j in range(1, self.N+1):
			plt.annotate(j, (xy[j,0], xy[j,1]))
		#end for i
		plt.show()
		
		
	#end def
	
	def greedy1(self):
		nodes = list(zip(range(self.N+1), self.nodes))[1:]
		#print(nodes)
		starting_node = nodes[0]
		nodes = nodes[1:]
		
		buses = []
		values = []
		capacities = []
		
		while len(nodes) > 0:
			current_bus = []
			current_value = 0
			current_capacity = self.capacity
			trash = []
			
			current_bus.append(starting_node[0])
			current_node = starting_node
			
			for node in nodes:
				index_min = min(range(len(nodes)), key=lambda i: 
					self.metric(current_node[1][0], nodes[i][1][0])
					#self.metric(current_node[1][0], nodes[i][1][0])/nodes[i][1][1]
				)
				#print(index_min)
				
				next_node = nodes[index_min]
				#print(next_node)
				
				if next_node[1][1] <= current_capacity:
					current_bus.append(nodes[index_min][0])
					#print(current_bus)
					current_value += self.metric(current_node[1][0], next_node[1][0])
					#print(current_value)
					current_capacity -= next_node[1][1]
					#print(current_capacity)
					current_node = next_node
				else:
					#trash.append(next_node)
					break
					#print(trash)
				#end if
				
				nodes = nodes[:index_min] + nodes[index_min+1:]
				#print(nodes)
			#end for
			
			#print(current_node)
			current_value += self.metric(current_node[1][0], starting_node[1][0])
			
			buses.append(current_bus)
			values.append(current_value)
			capacities.append(self.capacity - current_capacity)
			
			nodes += trash
			#print([i for (i, node) in nodes])
		#end while
		
		print(buses)
		print(values)
		print(capacities)
		
		print()
		print("(", end="")
		print(len(buses), end=", ")
		print(sum(values), end=")\n")
		
		#distances = [self.metric(self.nodes[self.depots[0]][0], coord) for (coord, demand) in self.nodes]
		return buses
	#end def
	
	def greedy2(self):
		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# SVE ISTO KAO GREEDY1, SAMO JOŠ PROVEDI TSP NA DOBIVENIM NODE-OVIMA
		nodes = list(zip(range(self.N+1), self.nodes))[1:]
		#print(nodes)
		starting_node = nodes[0]
		nodes = nodes[1:]
		
		buses = []
		values = []
		capacities = []
		
		while len(nodes) > 0:
			current_bus = []
			current_value = 0
			current_capacity = self.capacity
			trash = []
			
			current_bus.append(starting_node[0])
			current_node = starting_node
			
			for node in nodes:
				index_min = min(range(len(nodes)), key=lambda i: 
					self.metric(current_node[1][0], nodes[i][1][0])
					#self.metric(current_node[1][0], nodes[i][1][0])/nodes[i][1][1]
				)
				#print(index_min)
				
				next_node = nodes[index_min]
				#print(next_node)
				
				if next_node[1][1] <= current_capacity:
					current_bus.append(nodes[index_min][0])
					#print(current_bus)
					current_value += self.metric(current_node[1][0], next_node[1][0])
					#print(current_value)
					current_capacity -= next_node[1][1]
					#print(current_capacity)
					current_node = next_node
				else:
					#trash.append(next_node)
					break
					#print(trash)
				#end if
				
				nodes = nodes[:index_min] + nodes[index_min+1:]
				#print(nodes)
			#end for
			
			#print(current_node)
			current_value += self.metric(current_node[1][0], starting_node[1][0])
			
			buses.append(current_bus)
			values.append(current_value)
			capacities.append(self.capacity - current_capacity)
			
			nodes += trash
			#print([i for (i, node) in nodes])
		#end while
		
		print(buses)
		print(values)
		print(capacities)
		
		print()
		print("(", end="")
		print(len(buses), end=", ")
		print(sum(values), end=")\n")
		
		#distances = [self.metric(self.nodes[self.depots[0]][0], coord) for (coord, demand) in self.nodes]
		return buses
	#end def
#end class definition


if __name__ == "__main__":
	primjer = ProblemInstance("./instance/A-n32-k5.vrp")
	buses = primjer.greedy1()
	print(primjer.optimal_solution, end="\n\n")
	primjer.draw2d(buses)
	
	"""
	nodes = primjer.nodes
	probaj = [1,31,17,13,2,8,27,1]
	total = 0
	for i in range(len(probaj)-1):
		t1 = nodes[probaj[i]][0]
		t2 = nodes[probaj[i+1]][0]
		print(t1, t2)
		total += primjer.metric(t1,t2)
	#end for i
	print("new total: %f"%(total))
	"""
	
	
#end main
