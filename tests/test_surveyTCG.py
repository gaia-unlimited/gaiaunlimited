from gaiaunlimited.selectionfunctions.surveyTCG import DR3SelectionFunctionTCG_hpx7


def test_tcg():

    try:
        DR3SelectionFunctionTCG_hpx7()
    except:
        assert False, "Failed"
