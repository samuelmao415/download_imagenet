import requests, sys, os
import random, math
import pandas as pd
from PIL import Image
import socket

# loop through the list and download images
def download_imagenet(list,filedir,imagename):
    order=0
    for i in list:
        order=order+1
        filename = filedir+imagename+str(order)+'.jpg'
        try:
            session = requests.Session()
            session.max_redirects = 10 #this limits the maximum number of redirects
            try:
                imageConnection = session.get(i, timeout=(5, 14)) # get response from link with time out option: connect 5s, read 14s
                print(imageConnection.url)
                if 'photo_unavailable.png' in imageConnection.url: #dont download empty images from image net
                    print(str(order)+' is a blank image')
                    continue
                elif imageConnection.status_code != "404": # continue if image link is valid
                    with open(filename,'wb') as f:
                        f.write(imageConnection.content) # write image
                        print('downloaded imgae '+str(order))
                    try:
                        img = Image.open(filename).convert("RGB") # convert image (jpg/webp) to RGB
                        img.save(filename, "jpeg") # overwrite saved image
                        print('converted '+str(order)+' to RGB')
                    except IOError:
                        os.remove(filename)
                        print(str(i)+" failed to convert")
                        continue
                    print("---- Downloaded "+str(order/len(list))+' ------- ')
                else: # skip and log if image link is invalid
                    print(str(i)+" is invalid!")
                    continue
            #except requests.exceptions.TooManyRedirects or requests.exceptions.MissingSchema or http.client.IncompleteRead or requests.exceptions.Timeout: #avoid giving redirects instruction
            except requests.exceptions.RequestException or http.client.IncompleteRead as e:
                print(e)
                continue

        except requests.exceptions.ConnectionError: #continue if the request is blocked
            print('!request failed!'+str(order))
            continue
    print("job all finished")



text_file = open("headphone.txt", "r",encoding='utf8')
list = text_file.read().split('\n')
text_file.close()
filedir='./headphone/'
imagename='headphone'
list=list[330:335]
download_imagenet(list,filedir,imagename)
