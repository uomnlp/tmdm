import csv


def tsv_writer(docs, out_file="system.tsv", prefix=None):
	with open(out_file, 'w') as tsv:
		t_writer = csv.writer(tsv, delimiter='\t')
		for i in range(len(docs)):
			for ne in docs[i]._.nes:
				tsvrow = []
				tsvrow.append(str(i)) # Document id might become the long thing in current pipeline instead of just 0 to 20
				tsvrow.append(ne.start_char)
				tsvrow.append(ne.end_char)

				kb_url = ne._.ne_meta.get('uri')
				if kb_url:
					kb_id = kb_url.split('/')[-1]
					if prefix:
						kb_url = f"{prefix}/{kb_id}"
				else:
					kb_url = "nil"
				tsvrow.append(kb_url)

				tsvrow.append(1)

				if ne.label_ == "PER":
					label = "Person"
				elif ne.label_ == "ORG":
					label = "Organisation"
				elif ne.label_ == "LOC":
					label = "Location"
				elif ne.label_ == "DATE":
					label = "Date"
				elif ne.label_ == "MISC":
					label = "Misc"
				else:
					label = ne.label_
				tsvrow.append(label)

				t_writer.writerow(tsvrow)