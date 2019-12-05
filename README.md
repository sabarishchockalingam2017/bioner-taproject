# bioner-taproject
Bio-NER text analytics course project to recognize and tag genes and protein from text.

Run api.py to start server for inputting text into NER model. 
Enter following command in a terminal window to get NER results for your text:
curl -X GET http://127.0.0.1:5000/ -d text='Your desired text here.'


To rebuild best model please run crf.py.