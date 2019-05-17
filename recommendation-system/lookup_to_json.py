import numpy as np 
import pandas as pd
import sys 

def lookup_to_json(arr,json_dir):
	n  = len(arr)
	arr = np.vstack((np.arange(n),arr))
	df = pd.DataFrame(arr , index=["index","Name"])
	df.to_json(json_dir)

if __name__=="__main__":
        analysis_type = sys.argv[1]
	arr_dir="/home/lwidowski/superstore_data/superstore_"+analysis_type+"/"
   	arr=np.load(arr_dir+"superstore_"+analysis_type+"_contentId"+".npy")
	lookup_to_json(arr,arr_dir+"superstore_"+analysis_type+"_contentId"+".json")
	arr=np.load(arr_dir+"superstore_"+analysis_type+"_clientId"+".npy")
	lookup_to_json(arr,arr_dir+"superstore_"+analysis_type+"_clientId"+".json")	

