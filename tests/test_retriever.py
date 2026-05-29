from src.app.services.retriever import build_metadata_filter


def test_build_metadata_filter_empty():
    assert build_metadata_filter(None) is None
    assert build_metadata_filter({}) is None


def test_build_metadata_filter_single_condition():
    result = build_metadata_filter(
        {
            "doc_type": "paper",
            "tag": None,
            "source": None,
        }
    )

    assert result == {
        "doc_type": {
            "$eq": "paper",
        }
    }


def test_build_metadata_filter_multiple_conditions():
    result = build_metadata_filter(
        {
            "doc_type": "paper",
            "tag": "RAG",
            "source": "upload",
        }
    )

    assert result == {
        "$and": [
            {
                "doc_type": {
                    "$eq": "paper",
                }
            },
            {
                "tag": {
                    "$eq": "RAG",
                }
            },
            {
                "source": {
                    "$eq": "upload",
                }
            },
        ]
    }


def test_build_metadata_filter_ignores_empty_string():
    result = build_metadata_filter(
        {
            "doc_type": "paper",
            "tag": "   ",
            "source": "",
        }
    )

    assert result == {
        "doc_type": {
            "$eq": "paper",
        }
    }