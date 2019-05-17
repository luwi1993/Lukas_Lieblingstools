import pandas as pd 
import numpy as np 
import sys 

def main(): 
  	dir_to_file = sys.argv[1]
	M,keys=get_data(dir_to_file+".csv")
	clients = M[:,0]
	items=M[:,1]
	n_row,n_col = np.shape(M)
	lookups = [[] for i in range(n_col-1)]
	for i in range(n_col-1):
		M[:,i],lookups[i]=unique_int_and_lookup(M[:,i])		 
	matrix_to_csv(M,dir_to_file+"_processed.csv")
	for n in range(len(lookups)):
		np.save(dir_to_file+"_"+keys[n],lookups[n])

def usage():
	print("This Funktion takes a table of strings and returns a table of ints and floats plus lookup tables")
	print("USE LIKE THIS: python preprocess.py dir_to_in_file->String dir_to_out_file->String")

def get_data(dir_to_file="../tensorflow-recommendation-wals/data/superstore_menge_unprocessed"):
	data =  pd.read_csv(dir_to_file)
	return data.values,data.keys()

def string_to_unique_int(arr):
	uniques = np.unique(arr)
	for i in range(len(arr)):
		arr[i]=np.argwhere(uniques == arr[i])[0][0]	
    	return arr

def get_lookups(arr):
	uniques = np.unique(arr)
	lookup = ["obj" for i in range(len(uniques))]
	for i in range(len(arr)):
		index = np.argwhere(uniques == arr[i])[0][0]
		lookup[index]=arr[i]	
    	return lookup

def unique_int_and_lookup(arr):
	uniques = np.unique(arr)
	lookup = ["obj" for i in range(len(uniques))]
	for i in range(len(arr)):
		index = np.argwhere(uniques == arr[i])[0][0]
		lookup[index]= arr[i]
		arr[i]=index
	return arr,lookup

def filter(M,chosen_columns=[]):
	ret = M[:,chosen_columns[0]]
	for i in np.delete(np.array(chosen_columns),0):
		ret=np.vstack((ret,M[:,i]))
	return ret.T

def matrix_to_csv(M,name_csv_file="test"):
	n_row,n_col = np.shape(M)
	csv_file = open(name_csv_file,"w")
	csv_file.write("clientId,contentId,timeOnPage\n")
	for row in range(n_row):
		for col in range(n_col):
			csv_file.write(str(M[row,col]))
			if col == n_col-1:
				csv_file.write("\n")
			else:
				csv_file.write(",")		
	csv_file.close()


if __name__ == "__main__":	
	main()
