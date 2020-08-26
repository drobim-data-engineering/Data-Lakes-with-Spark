import configparser
from datetime import datetime
import boto3
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType

# Define config_file
config_file = 'dl.cfg'

# Reading cfg file
config = configparser.ConfigParser()
config.read(config_file)

# Setting up Access Key and Secret Key
AWS_KEY = config.get('AWS','AWS_ACCESS_KEY_ID')
AWS_SECRET = config.get('AWS','AWS_SECRET_ACCESS_KEY')
AWS_REGION = config.get('AWS','REGION')

# Setting up Environment Variables
os.environ['AWS_ACCESS_KEY_ID'] = config.get('AWS','AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = config.get('AWS','AWS_SECRET_ACCESS_KEY')

# Define AWS Services
s3_client = boto3.client('s3', region_name=AWS_REGION, aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET)

def update_config_file(config_file, section, key, value):
    """Writes to an existing config file

    Args:
        config_file (ConfigParser object): [description]
        section (string): The section on the config file the user wants to write
        key (string): The key the user wants to write
        value (string): The value the user wants to write
    """
    try:
        # Reading cfg file
        config = configparser.ConfigParser()
        config.read(config_file)

        #Setting  Section, Key and Value to be write on the cfg file
        config.set(section, key, value)

        # Writting to cfg file
        with open(config_file, 'w') as f:
            config.write(f)
    except:
        print('ERROR')

def create_spark_session():
    """Creates an Spark Session based on the configuration required

    Returns:
        [spark.session]: An activate Spark Session
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark

def create_bucket(bucket_location, bucket_name):
    """Creates S3 Bucket based on AWS Region and Bucket name

    Args:
        bucket_location (string): AWS Region to create S3 Bucket
        bucket_name (string): S3 Bucket Name
    """
    try:
        location = {'LocationConstraint': bucket_location}
        s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
        print('S3 Bucket created: ', bucket_name)
    except:
        print('S3 Bucket already exists: ', bucket_name)

def process_song_data(spark, input_data, output_data):
    """Create songs and artists tables from song data stored on a S3 Bucket.
       The function loads the data from S3, process it into songs and artists tables, and write to partitioned parquet files on S3.

    Args:
        spark (spark.session): An active Spark session
        input_data (string): S3 bucket with input data
        output_data (string): S3 bucket to store output tables
    """

    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')

    # read song data file
    print('Reading Song data JSON Files: ' + str(datetime.now()))
    df = spark.read.json(song_data)
    print('Load of Song Data Files Completed: ' + str(datetime.now()))

    # extract columns to create songs table
    songs_table = df['song_id', 'title', 'artist_id', 'year', 'duration'].dropDuplicates()

    # write songs table to parquet files partitioned by year and artist
    songs_path = os.path.join(output_data, 'songs')

    print('Writing Songs Table To S3: ' + str(datetime.now()))
    songs_table.write.parquet(songs_path, mode='overwrite', partitionBy=["year", "artist_id"])
    print('Write of Songs Table To S3 Complete: ' + str(datetime.now()))

    # extract columns to create artists table
    artists_table = df['artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude'].dropDuplicates()

    # write artists table to parquet files
    artists_path = os.path.join(output_data, 'artists')

    print('Writing Artists Table to S3: ' + str(datetime.now()))
    artists_table.write.parquet(artists_path, mode='overwrite')
    print('Write of Artists Table to S3 Complete: ' + str(datetime.now()))

def process_log_data(spark, input_data, output_data):
    """Create users, time and songplay tables from log data stored on a S3 Bucket.
       The function loads the data from S3, process it into songs and artists tables, and write to partitioned parquet files on S3.

    Args:
        spark (spark.session): An active Spark session
        input_data (string): S3 bucket with input data
        output_data (string): S3 bucket to store output tables
    """

    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/*.json')

    # read log data file
    print('Loading Log Data: ' + str(datetime.now()))
    df = spark.read.json(log_data)
    print('Load of Log Data Complete: ' + str(datetime.now()))

    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table
    users_table = df['userId', 'firstName', 'lastName', 'gender', 'level'].dropDuplicates()

    # write users table to parquet files
    users_path = os.path.join(output_data, 'users')

    print('Writing Users Table to S3: ' + str(datetime.now()))
    users_table.write.parquet(users_path, mode='overwrite')
    print('Write of Users Table to S3 Complete: ' + str(datetime.now()))

    # create datetime column from timestamp column
    get_datetime = F.udf(lambda ts: datetime.fromtimestamp(ts // 1000), DateType())
    df = df.withColumn('datetime', get_datetime(df.ts))

    # extract columns to create time table
    time_table = df.select(
        F.col('datetime').alias('start_time'),
        F.hour('datetime').alias('hour'),
        F.dayofmonth('datetime').alias('day'),
        F.weekofyear('datetime').alias('week'),
        F.month('datetime').alias('month'),
        F.year('datetime').alias('year'),
        F.date_format('datetime', 'u').alias('weekday')
    ).dropDuplicates()

    # write time table to parquet files partitioned by year and month
    time_path = os.path.join(output_data, 'time')

    print('Writing Time Table to S3: ' + str(datetime.now()))
    time_table.write.parquet(time_path, mode='append', partitionBy=["year", "month"])
    print('Write of Time Table to S3 Complete: ' + str(datetime.now()))

    # read in song data to use for songplays table
    songs_path = os.path.join(output_data, 'songs')

    print('Loading Songs Table: ' + str(datetime.now()))
    song_df = spark.read.parquet(songs_path)
    print('Load of Songs Table Completed: ' + str(datetime.now()))

    # extract columns from joined song and log datasets to create songplays table
    df = df['datetime', 'userId', 'level', 'song', 'artist', 'sessionId', 'location', 'userAgent'].dropDuplicates()

    log_song_df = df.join(song_df, df.song == song_df.title)

    print('Creating Songplays Table: ' + str(datetime.now()))
    songplays_table = log_song_df.select(
        F.monotonically_increasing_id().alias('songplay_id'),
        F.col('datetime').alias('start_time'),
        F.year('datetime').alias('year'),
        F.month('datetime').alias('month'),
        F.col('userId').alias('user_id'),
        'level',
        'song_id',
        'artist_id',
        F.col('sessionId').alias('session_id'),
        'location',
        F.col('userAgent').alias('user_agent')
    ).dropDuplicates()

    # write songplays table to parquet files partitioned by year and month
    songplays_path = os.path.join(output_data, 'songplays')

    print('Writing Songplays Table to S3: ' + str(datetime.now()))
    songplays_table.write.parquet(songplays_path, mode='overwrite', partitionBy=["year", "month"])
    print('Write of Songplays Table to S3 Complete: ' + str(datetime.now()))

def main():

    # Initiate Spark Session
    print('Creating Spark Session: ' + str(datetime.now()))
    spark = create_spark_session()

    # Create S3 Bucket
    print("Creating S3 Buckets...")
    create_bucket(config.get('AWS', 'REGION'), config.get('S3', 'OUTPUT_BUCKET'))

    # Writing to .cfg file
    print('Updatting CFG file...')
    update_config_file(config_file, 'DATALAKE', 'OUTPUT_DATA', 's3a://' + config.get('S3', 'OUTPUT_BUCKET') + '/')

    # Setting Up S3 Buckets
    input_data = config.get('DATALAKE','INPUT_DATA')
    output_data = config.get('DATALAKE','OUTPUT_DATA')

    # Initiate Song Data Processing
    print('Song Data Processing Started: ' + str(datetime.now()))
    process_song_data(spark, input_data, output_data)
    print('Song Data Processing Completed: ' + str(datetime.now()))

    # Initiate log Data Processing
    print('Log Data Processing Started: ' + str(datetime.now()))
    process_log_data(spark, input_data, output_data)
    print('Log Data Processing Completed: ' + str(datetime.now()))

    print('ETL complete: ' + str(datetime.now()))

if __name__ == "__main__":
    main()