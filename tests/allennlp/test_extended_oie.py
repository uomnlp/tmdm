from tmdm.allennlp.oie import join_mwp


def test_join_mwp_joins_predicates():
    tags = "O B-ARG0 B-V O B-V B-ARG1 O".split()
    mask = [0, 0, 1, 1, 1, 0, 0]
    assert join_mwp(tags, mask) == "O B-ARG0 B-V I-V I-V B-ARG1 O".split()


def test_join_mwp_join_predicates_at_beginning():
    tags = "B-V O B-V B-ARG0 B-ARG1 O O".split()
    mask = [1, 1, 1, 0, 0, 0, 0]
    assert join_mwp(tags, mask) == "B-V I-V I-V B-ARG0 B-ARG1 O O".split()


def test_join_mwp_sets_new_arg_boundries():
    tags = "B-ARG0 B-V B-ARG1 I-ARG1 O O".split()
    mask = [0, 1, 1, 0, 0, 0]
    assert join_mwp(tags,mask) == "B-ARG0 B-V I-V B-ARG1 O O".split()


