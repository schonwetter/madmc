
import numpy as np
import math
from solver import *
from gurobipy import *

columns = ['Engine', 'Torque', 'Weight', 'Acceleration', 'Price', 'Pollution', 'Design', 'Frame']

def Create_Weighted_Sum_DM(X_p,weight=[]):
	if (len(weight)==0):
		weight = np.random.uniform(0,100,len(X_p.columns))
		weight=weight/weight.sum()
	max=-10000
	indice=0
	for i in range(len(X_p)):
		weightedsum=(weight*X_p[columns].iloc[i]).sum()
		if (weightedsum>max):
			max=weightedsum
			indice=i

	return(weight,indice)
	

def PMR(x,y,P):
	m=Model("PMR")
	weight=[]
	#create variables
	for i in range(len(columns)):
		weight.append(m.addVar(vtype=GRB.CONTINUOUS,name="x"+str(i),lb=0,ub=1))
		m.update()
	
	#create objective
	obj = LinExpr();
	obj =0
	obj = quicksum([weight[i]*y[columns[i]]-weight[i]*x[columns[i]] for i in range(len(columns)) ])
	m.setObjective(obj,GRB.MAXIMIZE)
	
	#create constraints
	
	m.addConstr(quicksum(weight)==1 )
	
	for pair in P:
		y=pair[0]
		x=pair[1]

		
		m.addConstr( quicksum([ weight[i]*(y[columns[i]]-x[columns[i]]) for i in range (len(columns)) ]) >=0)
	#m.write("output.lp")
	#solve and retrieve value
	m.setParam( 'OutputFlag', False )
	m.optimize()
	weightValue=[p.x for p in weight ]
	return m.objVal,weightValue
	
def MR(x,X_p,P):
	max=-10000
	current_y=-1
	current_w=[]
	for y in range (len(X_p)):
		value,weight=PMR(x,X_p.iloc[y],P)
		if (value>max):
			max=value
			current_y=y
			current_w=weight
	return max,current_y,current_w
	
def MMR(X_p,P):
	min=10000
	best_x=-1
	best_y=-1
	current_w=[]

	for x in range (len(X_p)):
		value,current_y,weight=MR(X_p.iloc[x],X_p,P)
		if (value<min):
			min=value
			best_x=x
			best_y=current_y
			current_w=weight

	return best_x,best_y,current_w,min
	


def fitness(weight,x):
	#print(weight)
	return (weight*x[columns]).sum()

def Best_Sol(weight,X_p):
	indice=-1
	max=-10000
	for i in range(len(X_p)):
		weightedsum=fitness(weight,X_p.iloc[i])
		if (weightedsum>max):
			max=weightedsum
			indice=i

	return indice
	
	

	
def CSS(X_p):
	weight_DM,indice_DM=Create_Weighted_Sum_DM(X_p)
	indice_CSS=-1
	weight_CSS=[]
	P=[]
	cpt_question=0
	while (indice_DM!= indice_CSS):
		#compute mmr
		x,y,weight_CSS,mmr=MMR(X_p,P)
		indice_CSS=x
		print("nombre de comparaison ="+str(len(P)))
		print("MMR="+str(mmr)+"!!!!!!!")
		
		print("indice de la voiture preferee:"+str(indice_DM))
		print("indice minimisant les regrets max:"+str(indice_CSS))
		
		if (indice_DM== indice_CSS):
			break
		print("On pose la question suivante voiture numero "+str(x)+" superieur a voiture numero "+str(y))
		
		#ask DM is preference between x and y
		if (fitness(weight_DM,X_p.iloc[x])>=fitness(weight_DM,X_p.iloc[y])):
			P.append((X_p.iloc[x],X_p.iloc[y]))
		else:
			P.append((X_p.iloc[y],X_p.iloc[x]))
		#print(P)
			
		
		
		
		cpt_question+=1
		

	return cpt_question
	
			
	
	
	
filename='data.csv'
columns_to_min = ['Weight', 'Acceleration', 'Price', 'Pollution']
# Read input data
X = pd.read_csv(filename, skiprows=1, dtype={'Design': np.float64, 'Frame': np.float64})
# Fill NaN values with mean
X.fillna(X.mean(), inplace=True)
# Numeric columns
columns = ['Engine', 'Torque', 'Weight', 'Acceleration', 'Price', 'Pollution', 'Design', 'Frame']
# min-max normalize
X_norm = (X[columns] - X[columns].min()) / (X[columns].max() - X[columns].min())
# Convert columns to minimize
X_norm[columns_to_min] = X_norm[columns_to_min].apply(lambda x: -x)
# Get pareto front
X_pareto = pareto_front(X_norm, use_cache=True)
#print(X_pareto)
X_pareto.rename( columns={'Unnamed: 0':'index'}, inplace=True )

X_p=X_pareto.loc[:, X_pareto.columns != 'index']
#weightequi=np.ones(8)
#weight,indice=Create_Weighted_Sum_DM(X_p,weight=weightequi)
weight,indice=Create_Weighted_Sum_DM(X_p)
i=X_pareto["index"][indice]

#		Test Create_Weighted_Sum_DM
#print(X.iloc[i])
#print(weight)

#		Test de PMR
#print(PMR(X_p.iloc[0],X_p.iloc[1],[]))
#print(X_p[columns].iloc[0],X_p[columns].iloc[1])

#print(X.iloc[X_pareto["index"][21]])

CSS(X_p)

