import spacy
import datetime
def import_model(model_name):
	print(f"Importing spacy model : {model_name}..")
	start = datetime.datetime.now()
	spacy.load(model_name,disable=['ner'])
	end = datetime.datetime.now()
	delta = end-start
	print(f"Imported model {model_name}")
	print(delta)

import_model("nl_core_news_md")
import_model("nl_core_news_lg")

