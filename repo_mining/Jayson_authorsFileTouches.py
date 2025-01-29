# Modified version of CollectFiles.py for getting author/date data
# Plan: Source files are defined as files containing code written in a language
    # Source files are determined by the file extension
        # .java (5 files), .kts (4 files), .kt (9 files), .cpp (1 file), .h (1 file), .txt (1 file)
import json
import requests
import csv

import os

# GitHub Authentication function - no reason to change this
def github_auth(url, lsttoken, ct):
    jsonData = None
    try:
        ct = ct % len(lstTokens)
        headers = {'Authorization': 'Bearer {}'.format(lsttoken[ct])}
        request = requests.get(url, headers=headers)
        jsonData = json.loads(request.content)
        ct += 1
    except Exception as e:
        pass
        print(e)
    return jsonData, ct


### Should be able to modify function code to use csv file - improve performance ###
# @dictFiles, empty dictionary of files
# @lstTokens, GitHub authentication tokens
# @repo, GitHub repo
def countfiles(dictfiles, lsttokens, repo):
    ipage = 1  # url page counter   ### Shouldn't need
    ct = 0  # token counter

    # Open csv file generated by CollectFiles.py to be read
    fileInput = 'data\file_rootbeer.csv'
    fileCSV = open(fileInput)
    reader = csv.reader(fileCSV)

    try:
        # loop though all the commit pages until the last returned empty page - Should be able to replace while loop with for loop on csv reader
        while True:
            # Shouldn't need this section
            spage = str(ipage)
            commitsUrl = 'https://api.github.com/repos/' + repo + '/commits?page=' + spage + '&per_page=100'
            jsonCommits, ct = github_auth(commitsUrl, lsttokens, ct)

            # jsonCommits contains author names and dates for multiple commits
            # print("jsonCommits:")
            # print(jsonCommits)

            # Testing way of using csv file
            # for filename, touches in reader:    # Only use filenames with appropriate extensions
            #     testUrl = 'https://api.github.com/repos/' + repo + '/contents/' + filename
            #     test, ct = github_auth(testUrl, lsttokens, ct)      # Get sha - for shaObject in test...

            # break out of the while loop if there are no more commits in the pages - Shouldn't need this
            if len(jsonCommits) == 0:
                break
            # iterate through the list of commits in  spage
            for shaObject in jsonCommits:   # use test instead of jsonCommits
                sha = shaObject['sha']
                # For each commit, use the GitHub commit API to extract the files touched by the commit
                shaUrl = 'https://api.github.com/repos/' + repo + '/commits/' + sha
                shaDetails, ct = github_auth(shaUrl, lsttokens, ct)

                # shaDetails contains info on a commit: author, data, files touched
                # print("shaDetails:")
                # print(shaDetails)
                # break

                filesjson = shaDetails['files']
                for filenameObj in filesjson:
                    filename = filenameObj['filename']

                    # Could get source files by checking extensions
                    # root, extension = os.path.splitext(file_path)

                    dictfiles[filename] = dictfiles.get(filename, 0) + 1
                    print(filename)
            ipage += 1
    except:
        print("Error receiving data")
        exit(0)
# GitHub repo
repo = 'scottyab/rootbeer'

# Remember to delete token
lstTokens = []
                #"fd02a694b606c4120b8ca7bbe7ce29229376ee",
                #"16ce529bdb32263fb90a392d38b5f53c7ecb6b",
                #"8cea5715051869e98044f38b60fe897b350d4a"]

# Used for finding source codes; bit of a hard-coded solution though
extensions = ['.java', '.kts', '.kt', '.cpp', '.h', '.txt']

dictfiles = dict()
countfiles(dictfiles, lstTokens, repo)
print('Total number of source files: ' + str(len(dictfiles)))

### Shouldn't need anything below this ###
# file = repo.split('/')[1]
# # change this to the path of your file
# fileOutput = 'data/file_' + file + '.csv'
# rows = ["Filename", "Touches"]
# fileCSV = open(fileOutput, 'w')
# writer = csv.writer(fileCSV)
# writer.writerow(rows)

# bigcount = None
# bigfilename = None
# for filename, count in dictfiles.items():
#     rows = [filename, count]
#     writer.writerow(rows)
#     if bigcount is None or count > bigcount:
#         bigcount = count
#         bigfilename = filename
# fileCSV.close()
# print('The file ' + bigfilename + ' has been touched ' + str(bigcount) + ' times.')
