import os
# Function to rename multiple files. Used to rename large sets of images for curating the dataset.
def main():
   i = 0
   # insert your path
   path="E:/dataset/Filter_nsfw/temp_dataset/safe/"
   for filename in os.listdir(path):
      my_dest = "safe" + str(i) + ".jpg"
      my_source = path + filename
      my_dest = path + my_dest
      # rename() function will
      # rename all the files
      os.rename(my_source, my_dest)
      i += 1
# Driver Code
if __name__ == '__main__':
   # Calling main() function
   main()