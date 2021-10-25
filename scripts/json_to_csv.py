import json
import os
import pandas as pd





def json_to_csv():
    rootDir = '../jsons'
    jsonList = []
    jsonListToCSV = []
    tmpList = []

    for subdir, dirs, files in os.walk(rootDir):
        for jsonFile in files:
            if(jsonFile.endswith('.json')):
                jsonList = []


def main():
    print("Converting JSON files to CSV")
    json_to_csv()
    print("...Done")


if __name__== "__main__":
  main()