import paz


def test_imdb_labels_follow_released_weights_order():
    # The released weights are oarriaga/face_classification's
    # gender_mini_XCEPTION, whose src/utils/datasets.py get_labels("imdb")
    # is {0: "woman", 1: "man"} (IMDB-WIKI gender: 0 female, 1 male).
    assert paz.datasets.labels("IMDB") == ["woman", "man"]
