import os


# Function to rename multiple files. Used to rename large sets of images for curating the dataset.
def main():
    i = 0
    # insert the path to your dataset
    path = "E:/dataset/Filter_nsfw/temp_dataset/safe/"
    # using listdir from os to iterate on all the files
    for filename in os.listdir(path):
        my_dest = "safe" + str(i) + ".jpg"
        my_source = path + filename
        my_dest = path + my_dest
        # rename() function will
        # rename all the files
        os.rename(my_source, my_dest)
        i += 1


# Driver Code for the script
if __name__ == '__main__':
    # Calling main() function
    main()
