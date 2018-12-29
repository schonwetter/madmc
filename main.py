from solver import *
import solverPart2
    

def ask_answer(filename):
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
    
    ipt = get_ideal_point(X_p)

    npt = get_nadir_point(X_p)
    
    
    i = get_mindist_point(X_p, ipt, npt)

    indice=int(X_pareto.iloc[i]['index'])

    
    user="n"
    while(user=="n"):
    
        print("Here is the solution:")
        print(X.iloc[indice])
        user= raw_input("Are you satisfied of the solution found (y/n) ")
        
        if (user=="n") :
            for i in range (len(columns)):
                 print str(i) +":"+str(columns[i])

            obj= int(raw_input("Choose what you want to improve (number) "))
            
            #print(X['Name'],"  ",X[columns[obj]])
            
            print "What is your minimum/maximun acceptable value for ", columns[obj],"?"
            value=int(raw_input())
            
            #print(X_pareto)
            X_pareto=reject_solution_pareto(X,X_pareto,columns,columns_to_min,obj,value)
            
            #print(X_pareto)
            
            if (X_pareto.empty):
                print("You are too picky, no solutions found")
                return
            
            X_p=X_pareto.loc[:, X_pareto.columns != 'index']
            
            ipt = get_ideal_point(X_p)

            npt = get_nadir_point(X_p)
    
    
            i = get_mindist_point(X_p, ipt, npt)

            indice=int(X_pareto.iloc[i]['index'])
            
            
            
            
        


if __name__ == "__main__":
    
    
    ask_answer('data.csv')
    
    
    
