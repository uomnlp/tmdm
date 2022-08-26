import csv


def tsv_writer(docs, out_file="system.tsv", prefix=None):
	# This is useful for evaluating NER and EL with the neleval tool
	# Format is one row per NE: <doc id>    <ne start>    <ne end>    <el id>    1    <ne type>

	with open(out_file, 'w') as tsv:
		t_writer = csv.writer(tsv, delimiter='\t')
		for i in range(len(docs)):
			for ne in docs[i]._.nes:
				tsvrow = []
				# Document IDs are numbers 0 to number of docs
				tsvrow.append(str(i))

				tsvrow.append(ne.start_char)
				tsvrow.append(ne.end_char)

				# Prefix is meant for the EL ID, if none then has to be 'NIL' for neleval
				kb_url = ne._.ne_meta.get('uri')
				if kb_url:
					kb_id = kb_url.split('/')[-1]
					if prefix:
						kb_url = f"{prefix}/{kb_id}"
				else:
					kb_url = "NIL"
				tsvrow.append(kb_url)

				# Meant to be the confidence score but not needed
				tsvrow.append(1)

				# Formatting types to be the same as gold-standard .ann files
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