import pytest
from pyeCAP.acquisition_io.tdt_io import detect_stim_stores

def test_detect_stim_stores_from_storeslisting():
    # Simulate what you pasted from StoresListing.txt
    all_store_ids = [
        "251p", "251r", "AvET", "AvgE", "AvgG", "AvLT", "EMGt", "LIFt", "Pu2/", "RawE", "RawG"
    ]
    gizmo_name_map = {
        "251p": "Electrical Stim Driver",
        "251r": "Electrical Stim Driver",
        "AvET": "avgEMGTrain",
        "AvgE": "AvgLIFE",
        "AvgG": "AvgEMG",
        "AvLT": "avgLIFETrain",
        "EMGt": "LIFE_Filt",
        "LIFt": "LIFE_Filt",
        "Pu2/": "Pulse Generator",
        "RawE": "RawLIFE",
        "RawG": "RawEMG",
    }

    params, raws = detect_stim_stores(all_store_ids, gizmo_name_map)

    assert "251p" in params
    assert "251r" in raws
    # Make sure non-stim stores are not misclassified
    assert not {"RawE","RawG","AvET","AvgE","AvgG","AvLT"} & set(params)
    assert not {"AvET","AvgE","AvgG","AvLT"} & set(raws)

def test_detect_stim_stores_fallback_heuristic():
    # No gizmo map provided — rely on *p/*r suffix heuristic
    all_store_ids = ["251p", "251r", "RawE", "RawG"]
    params, raws = detect_stim_stores(all_store_ids, gizmo_name_map=None)

    assert params == ["251p"]
    assert "251r" in raws
