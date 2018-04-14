# Directories to read and write network files for supervised
# and unsupervised methods
#============================================================
directory_supervised = "Networks/"
directory_unsupervised = "Networks2/"

# Indicate if you are reading network edges from database or
# from file
#============================================================
from_file = 0

# Feature Selection
# Select features which should be used in classification phase
#============================================================
# AdamicAdar Jaccard CommonNeighbors PreferentialAttachment
# ShortestPath IN-DEGREE2 OUT-DEGREE2 PAGERANK2 BETWEENNESS
# Daily Rate
activeFeatures = [1,1,1,1,1,0,0,0,0,0,0,1,1]

# Feature importance function is activated by this flag
#============================================================
feature_importance = 0

# Set to one if there is a table in database called inactive_users
# and you dont want to consider them, set 0 otherwise
#============================================================
inactive_users_filter = 1

# Spark and Hadoop diretories
#============================================================
spark_home = ""
hadoop_home = ""


# OS Name
# There are some files that must be included if os is windows
#============================================================
osName = "WINDOWS" # Other options: MAC - LINUX


# Genelarization mode or normal mode
#============================================================
generalization = 0

# Time Series Forecasting Model
# average maverage wmaverage smoothing
# ============================================================
time_series = "wmaverage"
