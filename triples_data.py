
import requests

triples = [[
    ("Italy ", "Capital", "Rome"),
], [("France ", "Capital", "Paris"),
    ], [("Delhi ", "Capital", "India"),
    ], [("Berlin ", "Capital", "Germany"),
    ], [("Stockholm ", "Capital", "Sweden"),
    ], [("Vienna ", "Capital", "Austria"),
    ], [("Madrid ", "Capital", "Spain"),
    ]]



triples = [

]
def call_api():
    url = "http://167.86.123.100:8983/solr/vector-test-search/select?indent=true&q.op=OR&q=*%3A*&useParams="

    # Send a GET request
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse and print the response content
        data = response.json()

        for doc in data['response']['docs']:
            # atriples= map(lambda x: (x.split("|")[0], x.split("|")[1], x.split("|")[2]),  doc['triplePaths'])
            doc['triplePaths'] = list(set(doc['triplePaths'] ))
            triples.append(doc)
    else:
        print("Request failed with status code:", response.status_code)

