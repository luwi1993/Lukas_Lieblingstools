import numpy as np 
import pandas as pd
import sys 

def array_to_json(arr,json_dir):
	n_rows,n_cols = np.shape(arr)
	arr = np.vstack((np.arange(n_rows),arr.T))
	df = pd.DataFrame(arr,columns=["user_"+str(i) for i in range(n_rows)] , index=["index"]+["item_"+str(i) for i in range(n_cols)])
	df.to_json(json_dir)

def usage():
	print("USE LIKE THIS: python recs_to_json.py arr_dir->String json_dir->String") 

if __name__=="__main__":
	if sys.argv[1]=="usage":
		usage()
	else:
	        arr_dir = sys.argv[1]
		json_dir=sys.argv[2]
		arr=np.load(arr_dir)
		array_to_json(arr,json_dir)
	

