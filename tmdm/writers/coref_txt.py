def coref_txt_writer(docs, out_file="coref_output.txt"):
	# This is useful for evaluating OIE with Coreference Resolution with script
	# Format is one row per cluster: <doc_id>    <ent1_start>-<ent1_end>    <ent2_start>-<ent2_end>    ...

	with open(out_file, 'w') as txt:
		for i in range(len(docs)):
			clusters_seen = []
			for coref in docs[i]._.corefs:
				if coref.cluster_id in clusters_seen:
					continue

				onerow = str(i) + "\t"
				for ent in coref.cluster:
					onerow += str(ent.start_char) + "-" + str(ent.end_char) + "\t"

				onerow += "\n"
				txt.write(onerow)

				clusters_seen.append(coref.cluster_id)