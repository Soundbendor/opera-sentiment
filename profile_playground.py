from dataprofile import Profiler
from ENV import Data_PATH, Unified_PATH, Trimmed_PATH

using_PATH = Data_PATH
profiler = Profiler(using_PATH)
profiler.full_profile(if_print_profile=True)

# import matplotlib.pyplot as plt

# data = {'a-cappellas': 85, 'with music': 42}

# # Extract keys and values
# keys = list(data.keys())
# values = list(data.values())

# # Plot the distribution
# plt.bar(keys, values, color='skyblue')

# plt.xlabel('Recording Accompaniment')
# plt.ylabel('Count')
# plt.title('Distribution of Recording Accompaniment')
# # plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
# # plt.yticks(range(min(values), max(values) + 1))  # Set y-axis ticks to integer values

# for i in range(len(keys)):
#     plt.text(keys[i], values[i], str(values[i]), ha='center', va='bottom')
# plt.tight_layout()
# plt.show()
# plt.savefig('singing_accompaniment_distribution.png')