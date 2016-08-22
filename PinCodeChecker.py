
import csv
import json
import sys
from six.moves.urllib.request import urlopen
from nap.url import Url
import requests

class JsonApi(Url):
    def after_request(self, response):
        if response.status_code != 200:
            response.raise_for_status()
        return response.json()

def outputWriter( pin, response ):
    if(response is None):
        print("Bad response for Pin: " + pin)
        return;
    #jArray = json.loads( response )
    #if(jArray[0]['total'] == 0):
    if(response[0]['total'] == 0):
        print("No response for Pin: " + pin)
        return;
    with open('Addresses.csv', 'a') as csvfile:
        try:
            spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([pin, response[1]['v'], response[1]['i'], response[1]['lat'], response[1]['lon']])
            #spamwriter.writerow([pin, jArray[1]['v'], jArray[1]['i'], jArray[1]['lat'], jArray[1]['lon']])
            csvfile.flush()
            print("Added address for pin code : " + pin)
        except:
            print("Failure in pin: " + pin)
            pass

def evaluatePin( pin ):
    link = "http://www.streetdirectory.com/api/?mode=search&profile=sd_auto&country=sg&q=" + pin + "&output=json&v=1.0.1.798"
    try:
        api = JsonApi(link)
        #response = urlopen(link)
        response=api.get()
        #content = response.read().decode("utf-8")
        outputWriter( pin, response )
    except:
        print("URL Error for Pin: " + pin)

count = 10000
while (count < 84000):
    evaluatePin(str(count).zfill(6))
    #print ("The count is: " + str(count))
    count = count + 1

print ("Done!")
