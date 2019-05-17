
import sys
from google.cloud import bigquery

def usage():
	print("USE LIKE THIS: python load_to_bq.py dataset_id->String bucket_dir_to_json->Strng table_name->String") 

def to_bq(dataset_id,bucket_dir_to_json="gs://alessio-rs-bucket",table_name="us_states"):
	client = bigquery.Client()
	dataset_ref = client.dataset(dataset_id)
	job_config = bigquery.LoadJobConfig()
	job_config.autodetect = True
	job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
	uri = bucket_dir_to_json
	load_job = client.load_table_from_uri(
    	uri,
    	dataset_ref.table(table_name),
    	job_config=job_config)  # API request
	print('Starting job {}'.format(load_job.job_id))

	load_job.result()  # Waits for table load to complete.
	print('Job finished.')

	destination_table = client.get_table(dataset_ref.table(table_name))
	print('Loaded {} rows.'.format(destination_table.num_rows))

if __name__ == "__main__":
	if sys.argv[1]=="usage":
		usage()
	else:
		dataset_id = sys.argv[1]
		bucket_dir_to_json=sys.argv[2]
		table_name=sys.argv[3]
		to_bq(dataset_id,bucket_dir_to_json,table_name)
