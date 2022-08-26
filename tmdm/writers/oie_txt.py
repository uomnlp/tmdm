def oie_txt_writer(docs, out_file="oie_output.txt"):
	# This is useful for evaluating OIE with script
	# Format is one row per relation: <doc_id>    <verb_start>-<verb_end>   <arg1_start>-<arg1_end>    <arg2_start>-<arg2_end>    ...
	
	with open(out_file, 'w') as txt:
		for i in range(len(docs)):
			for oie in docs[i]._.oies:
				onerow = str(i) + "\t" + str(oie.start_char) + "-" + str(oie.end_char) + "\t"
				for arg in oie.arguments:
					onerow += str(arg.start_char) + "-" + str(arg.end_char) + "\t"
				
				onerow += "\n"
				txt.write(onerow)