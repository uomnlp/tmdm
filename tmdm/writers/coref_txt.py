def coref_txt_writer(docs, out_file="coref_output.txt"):
	with open(out_file, 'w') as txt:
		for i in range(len(docs)):
			clusters_seen = []
			for coref in docs[i]._.corefs:
				# One line per cluster
				if coref.cluster_id in clusters_seen:
					continue

				# Format: <doc_id>    <ent1_start>-<ent1_end>    <ent2_start>-<ent2_end>    ...
				onerow = str(i) + "\t"
				for ent in coref.cluster:
					onerow += str(ent.start_char) + "-" + str(ent.end_char) + "\t"

				onerow += "\n"
				txt.write(onerow)

				clusters_seen.append(coref.cluster_id)